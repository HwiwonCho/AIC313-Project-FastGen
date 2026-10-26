"""Measure an exported model with Yunseo's scaled-branch TRAIN755 protocol.

This is a separate comparable metric, not the official validation3020 score.
Use the same immutable dataset contents and peer manifest to verify reference IDs.
"""
import argparse
import json
from pathlib import Path

from fastgen.data import prepare_manifest
from fastgen.metrics import train_proxy
from fastgen.runtime import atomic_json, digest
from model import Model


def compare_references(ours, peer):
    def rows(manifest):
        return [{key: row[key] for key in ("path", "category", "group")} for row in manifest["references"]]
    if rows(ours) != rows(peer):
        raise ValueError("Reference IDs/order/pixel groups differ from the supplied peer manifest")
    return digest(rows(ours))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--mode", choices=["one_nfe", "few_nfe"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split-dir", type=Path, default=Path(__file__).parent/"data/pokemon-generation-one-22k")
    p.add_argument("--peer-manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = prepare_manifest(a.data_root, a.split_dir, a.output/"manifest.json")
    reference_verification = "code defaults matched; actual remote overrides not verified"
    if a.peer_manifest:
        compare_references(manifest, json.loads(a.peer_manifest.read_text()))
        reference_verification = "reference IDs/order/pixel hashes match supplied peer manifest"
    model = Model.load_checkpoint(str(a.checkpoint), a.mode, device=a.device)
    result = train_proxy(model.backbone, manifest, a.data_root, a.output/"evaluation", device=a.device,
                method="endpoint1" if a.mode=="one_nfe" else "renoise4", batch_size=8,
                precision="fp32", seed=27182, workers=4, reference_overlaps_training=True)
    result["peer_reference_verification"] = reference_verification
    atomic_json(a.output/"peer_fid.json", result)
    print(json.dumps(result, indent=2))
