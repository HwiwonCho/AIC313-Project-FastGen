import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline
from fastgen import orchestration
from fastgen import engine
from fastgen.data import PairBatches, TrainBatches, prepare_manifest
from fastgen.metrics import peer_evaluation_precision, train_proxy
from fastgen.pairs import generate_pairs
from fastgen.runtime import MemoryGuard, atomic_json, atomic_save, digest, sha256
from fastgen.sampling import sample_images
from model import Model, ModelFewNFE, ModelOneNFE
from peer_fid import compare_references
from submission_check import check_submission

TINY = {"num_categories": 2, "base_channels": 8, "channel_mult": [1, 2], "num_blocks": 1}
torch.set_num_threads(2)


class Fixture(unittest.TestCase):
    def setUp(self):
        scratch = ROOT/".test_tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)
        self.data = self.root/"data"
        self.split = self.root/"split"
        self.split.mkdir()
        paths = []
        for category, name in enumerate(("A", "B")):
            for i in range(8):
                relative = f"PokemonData/{name}/{i}.png"
                path = self.data/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                # First two files are a within-class exact pixel duplicate.
                color_index = max(0, i-1)
                Image.new("RGB", (72, 70), (category*100+color_index*7, 30+color_index*5, 180-color_index)).save(path)
                paths.append(relative)
        (self.split/"train_split.txt").write_text("\n".join(paths)+"\n")
        atomic_json(self.split/"category_to_id.json", {"A": 0, "B": 1})
        self.manifest_path = self.root/"manifest.json"
        self.manifest = prepare_manifest(self.data, self.split, self.manifest_path,
              holdout_per_class=2, expected_classes=2, expected_train_files=16)

    def tearDown(self):
        self.temp.cleanup()

    def task(self, name, kind="soflow", source=None, pairs=None):
        return {"name": name, "run_dir": str(self.root/name), "manifest": str(self.manifest_path),
                "data_root": str(self.data), "split_dir": str(self.split), "device": "cpu",
                "precision": "fp32", "architecture": TINY, "vram_limit_bytes": 20_000_000_000,
                "minimum_free_bytes": 0, "eval_batch_size": 2,
                "evaluation": {"precision": "fp32", "seed": 27182, "workers": 0},
                "code_sha256": "synthetic_cpu_fixture", "source": str(source) if source else None,
                "pairs": str(pairs) if pairs else None,
                "settings": {"kind": kind, "steps": 4, "microbatch": 4, "accumulation": 1,
                    "pilot": False, "augmentation": "flip", "unique_sampling": False,
                    "seed": 42, "lr": 1e-4 if kind=="soflow" else 1e-5,
                    "horizon": 4, "warmup": 1, "eval_every": 4, "save_every": 1,
                    "log_every": 4, "sampling_method": "endpoint1" if kind=="distill" else "renoise4"}}

    @staticmethod
    def fake_proxy(*args, **kwargs):
        # A unit-test stub, never a production FID or model-quality claim.
        return {"FID_train_proxy": 100., "synthetic_test_metric": True}

    def test_group_holdout_and_paired_augmentation_draws(self):
        excluded = {r["group"] for r in self.manifest["references"]}
        self.assertTrue(all(r["group"] not in excluded for r in self.manifest["pilot_rows"]))
        a = TrainBatches(self.data, self.manifest, pilot=True, augmentation="flip")
        b = TrainBatches(self.data, self.manifest, pilot=True, augmentation="spatial")
        x, y = a.batch(12, "cpu")
        xx, yy = b.batch(12, "cpu")
        self.assertTrue(torch.equal(y, yy))
        self.assertEqual(a.state_dict(), b.state_dict())
        self.assertFalse(torch.equal(x, xx))
        # There is deliberately no val_split.txt; training preparation must not need it.
        self.assertFalse((self.split/"val_split.txt").exists())

    def test_complete_update_resume_matches_uninterrupted(self):
        uninterrupted = self.task("continuous")
        engine.train_stage(uninterrupted, proxy_function=self.fake_proxy)
        resumed = self.task("resumed")
        original = engine.soflow_objective
        def stop_at_second(model, images, labels, step, settings):
            result = original(model, images, labels, step, settings)
            if step == 1:
                signal.raise_signal(signal.SIGTERM)
            return result
        with patch.object(engine, "soflow_objective", stop_at_second):
            result = engine.train_stage(resumed, proxy_function=self.fake_proxy)
        self.assertEqual(result["phase"], "stopped")
        self.assertEqual(result["step"], 2)
        engine.train_stage(resumed, proxy_function=self.fake_proxy)
        a = torch.load(self.root/"continuous/current.pt", weights_only=False)
        b = torch.load(self.root/"resumed/current.pt", weights_only=False)
        for section in ("model", "ema"):
            self.assertTrue(all(torch.equal(a[section][k], b[section][k]) for k in a[section]))
        self.assertTrue(torch.equal(a["rng"]["cpu"], b["rng"]["cpu"]))
        self.assertEqual(a["batch_state"], b["batch_state"])

    def test_teacher_pairs_distillation_and_strict_submission_loader(self):
        engine.train_stage(self.task("teacher"), proxy_function=self.fake_proxy)
        source = self.root/"teacher/best.pt"
        pair_task = {"source": str(source), "output": str(self.root/"pairs"),
            "manifest": str(self.manifest_path), "split_dir": str(self.split),
            "method": "solution4", "pairs_per_class": 3, "batch_size": 2, "shard_size": 4,
            "seed": 101, "precision": "fp32", "code_sha256": "synthetic_cpu_fixture",
            "device": "cpu", "vram_limit_bytes": 20_000_000_000}
        index = generate_pairs(pair_task)
        self.assertEqual(index["count"], 6)
        before = [sha256(self.root/"pairs"/r["file"]) for r in index["shards"]]
        self.assertEqual(generate_pairs(pair_task), index)
        self.assertEqual(before, [sha256(self.root/"pairs"/r["file"]) for r in index["shards"]])
        pair_batches = PairBatches(self.root/"pairs")
        original_load = torch.load
        with patch("fastgen.data.torch.load", wraps=original_load) as loads:
            pair_batches.batch(16, "cpu")
            self.assertEqual(loads.call_count, 1, "One microbatch must not load a shard for each image")
        reflow_task = self.task("reflow", "reflow", source, self.root/"pairs")
        reflow_task["settings"]["sampling_method"] = "euler4"
        engine.train_stage(reflow_task, proxy_function=self.fake_proxy)
        task = self.task("student", "distill", source, self.root/"pairs")
        engine.train_stage(task, proxy_function=self.fake_proxy)
        saved = torch.load(self.root/"student/best.pt", weights_only=False)
        checkpoint = self.root/"one.ckpt"
        atomic_save(checkpoint, {"state_dict": {"backbone."+k:v for k,v in saved["ema"].items()}})
        model = Model.load_checkpoint(str(checkpoint), "one_nfe", device="cpu", **TINY)
        calls = []
        handle = model.backbone.network.register_forward_hook(lambda *x:calls.append(1))
        try:
            output = model.sample((2,3,64,64), device="cpu", category=torch.tensor([0,1]))
        finally:
            handle.remove()
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.isfinite(output).all())
        # Changed teacher bytes must not silently reuse an existing pair pool.
        bad = dict(pair_task, seed=102)
        with self.assertRaises(ValueError):
            generate_pairs(bad)

    def test_prefetch_preserves_augmented_batches_and_consumed_resume_state(self):
        from fastgen.prefetch import PrefetchBatches
        plain = TrainBatches(self.data, self.manifest, pilot=False, augmentation="spatial", seed=42)
        ahead = PrefetchBatches(TrainBatches(self.data, self.manifest, pilot=False, augmentation="spatial", seed=42), 4)
        try:
            for _ in range(3):
                expected = plain.batch(4, "cpu")
                actual = ahead.batch(4, "cpu")
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,actual)))
                self.assertEqual(plain.state_dict(), ahead.state_dict())
            consumed = ahead.state_dict()
        finally:
            ahead.close()
        resumed = PrefetchBatches(TrainBatches(self.data, self.manifest, pilot=False, augmentation="spatial", seed=42),4)
        resumed.load_state_dict(consumed)
        try:
            expected, actual = plain.batch(4,"cpu"), resumed.batch(4,"cpu")
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,actual)))
        finally:
            resumed.close()

    def test_prefetch_training_and_interrupted_full_state_resume_match(self):
        plain = self.task("plain_prefetch_control")
        ahead = self.task("prefetch_continuous")
        ahead["execution"] = {"soflow_prefetch_batches":2}
        engine.train_stage(plain, proxy_function=self.fake_proxy)
        engine.train_stage(ahead, proxy_function=self.fake_proxy)
        resumed = self.task("prefetch_resumed")
        resumed["execution"] = {"soflow_prefetch_batches":2}
        original = engine.soflow_objective
        def stop_at_second(model,images,labels,step,settings):
            result = original(model,images,labels,step,settings)
            if step==1:
                signal.raise_signal(signal.SIGTERM)
            return result
        with patch.object(engine,"soflow_objective",stop_at_second):
            result = engine.train_stage(resumed,proxy_function=self.fake_proxy)
        self.assertEqual(result["step"],2)
        engine.train_stage(resumed,proxy_function=self.fake_proxy)
        checkpoints=[torch.load(self.root/name/"current.pt",weights_only=False) for name in
                     ("plain_prefetch_control","prefetch_continuous","prefetch_resumed")]
        for expected in checkpoints[1:]:
            for section in ("model","ema"):
                self.assertTrue(all(torch.equal(checkpoints[0][section][k],expected[section][k]) for k in expected[section]))
            self.assertTrue(torch.equal(checkpoints[0]["rng"]["cpu"],expected["rng"]["cpu"]))
            self.assertEqual(checkpoints[0]["batch_state"],expected["batch_state"])

    def test_default_architecture_count_and_one_four_actual_calls(self):
        with torch.device("meta"):
            full = ModelFewNFE(device="meta")
        self.assertEqual(full.count_parameters(), 95_524_611)
        del full
        for cls, count in ((ModelOneNFE,1), (ModelFewNFE,4)):
            model = cls(device="cpu", **TINY).eval()
            calls = []
            hook = model.backbone.network.register_forward_hook(lambda *x:calls.append(1))
            try:
                result = model.sample((2,3,64,64), device="cpu", category=torch.tensor([0,1]))
            finally:
                hook.remove()
            self.assertEqual(len(calls), count)
            self.assertTrue((result.abs()<=1).all())
            prediction = model(torch.zeros(2,3,64,64), torch.ones(2), category=torch.tensor([0,1]))
            self.assertEqual(prediction.shape, (2,3,64,64))

    def test_execution_only_continuation_preserves_full_state_and_next_updates(self):
        from fastgen.continuation import fingerprint, migrate_checkpoint
        control, old = self.task("continuation_control"), self.task("continuation_old")
        engine.train_stage(control, proxy_function=self.fake_proxy)
        original = engine.soflow_objective
        def stop_at_second(model, images, labels, step, settings):
            result = original(model, images, labels, step, settings)
            if step == 1:
                signal.raise_signal(signal.SIGTERM)
            return result
        with patch.object(engine, "soflow_objective", stop_at_second):
            result = engine.train_stage(old, proxy_function=self.fake_proxy)
        self.assertEqual(result["step"], 2)
        new = self.task("continuation_new")
        new["execution"] = {"soflow_prefetch_batches": 2}
        new["code_sha256"] = "execution_only_test_revision"
        checkpoint = self.root/"continuation_old/current.pt"
        original_hash = sha256(checkpoint)
        bad = copy.deepcopy(new)
        bad["settings"]["lr"] *= .5
        with self.assertRaisesRegex(ValueError, "learning/data/FID"):
            migrate_checkpoint(checkpoint,bad,self.root/"rejected/current.pt",self.root/"rejected/receipt.json")
        receipt = migrate_checkpoint(checkpoint,new,self.root/"continuation_new/current.pt",self.root/"receipt.json")
        self.assertEqual(receipt["step"], 2)
        self.assertEqual(receipt["full_state_fingerprints_before"], receipt["full_state_fingerprints_after"])
        self.assertEqual(sha256(checkpoint), original_hash)
        engine.train_stage(new, proxy_function=self.fake_proxy)
        a = torch.load(self.root/"continuation_control/current.pt",weights_only=False)
        b = torch.load(self.root/"continuation_new/current.pt",weights_only=False)
        for key in ("model","ema","optimizer","rng","batch_state","step"):
            self.assertEqual(fingerprint(a[key]),fingerprint(b[key]),key)

    def test_resume_after_interrupted_best_adoption(self):
        task = self.task("best_copy_interruption")
        with patch.object(engine.shutil, "copyfile", side_effect=OSError("synthetic interrupted best copy")):
            with self.assertRaises(OSError):
                engine.train_stage(task, proxy_function=self.fake_proxy)
        self.assertTrue((self.root/"best_copy_interruption/current.pt").exists())
        self.assertFalse((self.root/"best_copy_interruption/best.pt").exists())
        result = engine.train_stage(task, proxy_function=self.fake_proxy)
        self.assertEqual(result["phase"], "complete")
        self.assertTrue((self.root/"best_copy_interruption/best.pt").exists())

    def test_fixed_files_and_policy_settings(self):
        pipeline.check_official()
        config = json.loads((ROOT/"configs/deadline.json").read_text())
        pipeline.validate_config(config)
        config["evaluation"]["seed"] = 1
        with self.assertRaises(ValueError):
            pipeline.validate_config(config)

    def test_full_orchestration_bundles_both_modes_and_records(self):
        # Exercise the real controller, locks, source snapshot and packaging.
        # Worker numerical paths are covered separately; no GPU/FID runs here.
        config = json.loads((ROOT/"configs/deadline.json").read_text())
        config["minimum_free_bytes"] = 0
        cfg_path = self.root/"config.json"
        atomic_json(cfg_path,config)
        directory = self.root/"flow"
        order=[]
        def prepared(root,split,output,**kwargs):
            atomic_json(output,self.manifest)
            return self.manifest
        def worker(action,task,run):
            order.append(task["name"])
            if action=="_train":
                folder=Path(task["run_dir"]);folder.mkdir(parents=True,exist_ok=True)
                atomic_json(folder/"summary.json",{"phase":"complete","best_train_proxy_fid":99. if task["name"]=="pilot_spatial" else 100.})
                (folder/"best.pt").write_bytes(b'synthetic_worker_fixture')
            elif action=="_export":
                output=Path(task["output"]);output.parent.mkdir(parents=True,exist_ok=True)
                output.write_bytes(b'synthetic_export_fixture')
            elif action=="_evaluate":
                atomic_json(run/"official_evaluation"/(task["mode"]+".json"),{"synthetic_test_metric":True})
        args=types.SimpleNamespace(config=cfg_path,data_root=self.data,split_dir=self.split,run_root=directory)
        with patch("fastgen.data.prepare_manifest",prepared),patch.object(orchestration,"child",worker):
            orchestration.execute_pipeline(args)
        self.assertEqual(order,["preflight","pilot_flip","pilot_spatial","teacher","pairs_teacher","distill",
                                "export_few_nfe","export_one_nfe","official_one_nfe","official_few_nfe"])
        self.assertEqual(json.loads((directory/"augmentation_selection.json").read_text())["augmentation"],"flip")
        self.assertEqual(json.loads((directory/"status.json").read_text())["phase"],"complete")
        for name in ("model.py","fastgen/engine.py","fastgen/submission.py","run.sh","CITATIONS.md",
                     "records/writeup_facts.json","checkpoints/one_nfe.ckpt","checkpoints/few_nfe.ckpt"):
            self.assertTrue((directory/"submission"/name).is_file(),name)

    def test_strict_vram_threshold_without_gpu_execution(self):
        guard = MemoryGuard.__new__(MemoryGuard)
        guard.device, guard.limit = torch.device("cuda:0"), 20_000_000_000
        with patch("torch.cuda.max_memory_allocated", return_value=20_000_000_000), patch("torch.cuda.max_memory_reserved", return_value=20_000_000_000):
            with self.assertRaises(RuntimeError):
                guard.check()
            self.assertEqual(guard.snapshot()["peak_allocated_bytes"], 20_000_000_000)

    def test_full_size_export_and_packaged_one_four_inference(self):
        # Real 95.52M shapes and forwards, random CPU weights, no training/FID.
        # All synthetic checkpoints live only in the temporary test directory.
        model = engine.build_model()
        state = model.state_dict()
        stage = self.root/"synthetic_full.pt"
        submission = self.root/"packaged"
        for kind, mode in (("soflow","few_nfe"), ("distill","one_nfe")):
            atomic_save(stage, {"architecture": engine.DEFAULT_ARCHITECTURE,
                        "identity": {"settings": {"kind": kind}}, "ema": state})
            pipeline.export_ema(stage, submission/"checkpoints"/(mode+".ckpt"), mode)
            (submission/"checkpoints"/(mode+".provenance.json")).unlink()
            pipeline.export_ema(stage, submission/"checkpoints"/(mode+".ckpt"), mode)
        del state, model
        for name in pipeline.code_inventory():
            target = submission/name
            target.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copyfile(ROOT/name, target)
        env = {k:v for k,v in os.environ.items() if k != "PYTHONPATH"}
        result = subprocess.run([sys.executable,"-B","submission_check.py",".","--device","cpu"],
                                cwd=submission, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn("1 calls", result.stdout)
        self.assertIn("4 calls", result.stdout)

    def test_deterministic_distillation_teacher_and_tf32_restore(self):
        torch.manual_seed(41)
        model = engine.build_model(TINY).eval()
        noise = torch.randn(2,3,64,64)
        categories = torch.tensor([0,1])
        a = sample_images(model, noise.shape, categories, device="cpu", noise=noise, method="solution4",
                          extra_generator=torch.Generator().manual_seed(1))
        b = sample_images(model, noise.shape, categories, device="cpu", noise=noise, method="solution4",
                          extra_generator=torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(a,b))
        previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                    torch.backends.cudnn.benchmark)
        try:
            torch.backends.cuda.matmul.allow_tf32=True
            torch.backends.cudnn.benchmark=True
            with self.assertRaises(ValueError):
                with peer_evaluation_precision():
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertTrue(torch.backends.cudnn.allow_tf32)
                    self.assertFalse(torch.backends.cudnn.benchmark)
                    raise ValueError("synthetic metric failure")
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.benchmark)
        finally:
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark=previous

    @unittest.skipUnless(os.environ.get("FASTGEN_PEER_SOURCE"), "Optional actual teammate source parity check")
    def test_actual_peer_reference_selection_and_png_protocol(self):
        source = Path(os.environ["FASTGEN_PEER_SOURCE"])
        sys.path.insert(0, str(source))
        try:
            spec = importlib.util.spec_from_file_location("peer_train_for_test", source/"scaled_train.py")
            peer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(peer)
            their_manifest = peer.prepare(self.data, self.split, reference_per_class=2, seed=42)
            compare_references(self.manifest, their_manifest)
            class Dummy(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.tensor(0.))
                    self.num_categories = 2
                    self.config = types.SimpleNamespace(num_classes=2)
                    self.network = torch.nn.Identity()
                def solution(self, x, *args):
                    return self.network(x)
                def sample(self, shape, device, category, generator, **kwargs):
                    return torch.randn(shape, device=device, generator=generator).clamp(-1,1)
            dummy = Dummy().eval()
            args = types.SimpleNamespace(run_dir=self.root/"peer", data_root=self.data, session_id="fixture",
                    eval_seed=27182, eval_batch_size=8, selection_steps=1, eval_guidance=1, workers=4)
            args.run_dir.mkdir()
            dirs = []
            def fid_stub(generated, reference, **options):
                dirs.append((Path(generated), Path(reference), options))
                return 0.
            peer.evaluate_proxy(dummy, their_manifest, args, 1, fid_function=fid_stub)
            train_proxy(dummy, self.manifest, self.data, self.root/"ours", device="cpu", method="endpoint1",
                        batch_size=8, precision="fp32", seed=27182, workers=4, fid_function=fid_stub)
            for j in (0,1):
                self.assertEqual([sha256(p) for p in sorted(dirs[0][j].glob("*.png"))],
                                 [sha256(p) for p in sorted(dirs[1][j].glob("*.png"))])
            self.assertEqual(dirs[0][2], dirs[1][2])
        finally:
            sys.path.remove(str(source))


if __name__ == "__main__":
    unittest.main()
