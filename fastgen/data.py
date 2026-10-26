"""Read-only TRAIN preparation, group-safe pilot holdout and resumable batches.

Weak affine ranges follow the team's shared augmentation policy (see CITATIONS).
The official validation manifest is never opened by this module.
"""
import hashlib
import json
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .runtime import atomic_json, sha256


def image64(path):
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
        return TF.resize(rgb, [64, 64], interpolation=InterpolationMode.BILINEAR, antialias=True)


def safe_image(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid train image path")
    result = (root / relative).resolve(strict=True)
    if not result.is_relative_to(root):
        raise ValueError("Train image escapes the data root")
    return result


def prepare_manifest(root, split_dir, output, *, holdout_per_class=5, seed=42,
                     expected_classes=151, expected_train_files=17079):
    root, split_dir, output = Path(root), Path(split_dir), Path(output)
    mapping = json.loads((split_dir / "category_to_id.json").read_text(encoding="utf-8-sig"))
    paths = [x.strip() for x in (split_dir / "train_split.txt").read_text().splitlines() if x.strip()]
    if len(mapping) != expected_classes or sorted(mapping.values()) != list(range(expected_classes)):
        raise ValueError("Unexpected category mapping")
    if len(paths) != expected_train_files or len(paths) != len(set(paths)):
        raise ValueError("Unexpected train manifest")
    rows, groups = [], defaultdict(list)
    for i, relative in enumerate(paths):
        path = safe_image(root, relative)
        category = mapping[Path(relative).parent.name]
        group = hashlib.sha256(image64(path).tobytes()).hexdigest()
        row = {"path": relative, "category": category, "group": group}
        rows.append(row)
        groups[group].append(row)
        if (i + 1) % 2000 == 0:
            print(f"Train image audit: {i+1}/{len(paths)}", flush=True)
    eligible = defaultdict(list)
    for group, members in sorted(groups.items()):
        if len({m["category"] for m in members}) == 1:
            eligible[members[0]["category"]].append(members[0])
    rng, reference = random.Random(seed), []
    for category in range(expected_classes):
        candidates = list(eligible[category])
        if len(candidates) < holdout_per_class:
            raise ValueError(f"Class {category} has too few unique holdout groups")
        rng.shuffle(candidates)
        reference.extend(candidates[:holdout_per_class])
    excluded = {x["group"] for x in reference}
    result = {"version": 1, "seed": seed, "holdout_per_class": holdout_per_class,
              "mapping": mapping, "train_rows": rows, "references": reference,
              "pilot_rows": [x for x in rows if x["group"] not in excluded],
              "train_split_sha256": sha256(split_dir / "train_split.txt"),
              "mapping_sha256": sha256(split_dir / "category_to_id.json"),
              "train_files": len(rows), "unique_pixel_groups": len(groups),
              "pixel_definition": "white-composited bilinear64 RGB before augmentation",
              "pilot_reference_overlaps_training": False,
              "final_reference_overlaps_training": True,
              "cross_label_groups": sum(len({r["category"] for r in g}) > 1 for g in groups.values())}
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError("Prepared data changed; use a fresh run directory")
    else:
        atomic_json(output, result)
    return result


def read_manifest(path, split_dir):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, key in [("train_split.txt", "train_split_sha256"), ("category_to_id.json", "mapping_sha256")]:
        if sha256(Path(split_dir) / name) != value[key]:
            raise ValueError("Fixed train split/mapping changed")
    return value


class TrainBatches:
    """Class-balanced replacement; independent sampling and augmentation streams.

    Default sampling preserves original file weights, matching our prior recipe.
    Unique-group sampling is configurable but is not another default experiment.
    """
    def __init__(self, root, manifest, *, pilot, augmentation, seed=42, unique=False):
        if augmentation not in ("flip", "spatial"):
            raise ValueError("Invalid augmentation policy")
        self.root, self.augmentation = Path(root), augmentation
        self.rng, self.aug_rng = random.Random(seed), random.Random(seed + 1)
        self.by_class = defaultdict(list)
        seen = set()
        for row in manifest["pilot_rows" if pilot else "train_rows"]:
            key = row["category"], row["group"]
            if not unique or key not in seen:
                self.by_class[row["category"]].append(row)
                seen.add(key)
        self.classes = len(manifest["mapping"])
        if any(not self.by_class[c] for c in range(self.classes)):
            raise ValueError("A class has no training examples")
        self.cache, self.draws = OrderedDict(), 0

    def state_dict(self):
        return {"sampling_rng": self.rng.getstate(), "augmentation_rng": self.aug_rng.getstate(), "draws": self.draws}

    def load_state_dict(self, state):
        self.rng.setstate(state["sampling_rng"])
        self.aug_rng.setstate(state["augmentation_rng"])
        self.draws = state["draws"]

    def batch(self, count, device):
        images, labels = [], []
        for _ in range(count):
            label = self.rng.randrange(self.classes)
            row = self.rng.choice(self.by_class[label])
            relative = row["path"]
            if relative not in self.cache:
                self.cache[relative] = image64(safe_image(self.root, relative))
                if len(self.cache) > 2048:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(relative)
            image = self.cache[relative]
            # Draw both policies' parameters even in the flip-only control so
            # subsequent flip decisions and original image draws stay paired.
            flip = self.aug_rng.random() < .5
            angle = self.aug_rng.uniform(-5, 5)
            shift = [self.aug_rng.randint(-3, 3), self.aug_rng.randint(-3, 3)]
            scale = self.aug_rng.uniform(.9, 1.05)
            if flip:
                image = TF.hflip(image)
            if self.augmentation == "spatial":
                image = TF.affine(image, angle, shift, scale, [0., 0.],
                                  interpolation=InterpolationMode.BILINEAR, fill=[255, 255, 255])
            images.append(TF.to_tensor(image).mul(2).sub(1))
            labels.append(label)
        self.draws += count
        return torch.stack(images).to(device), torch.tensor(labels, device=device)


class PairBatches:
    def __init__(self, directory, seed=42):
        self.root = Path(directory)
        self.index = json.loads((self.root / "index.json").read_text())
        self.rng, self.cache = random.Random(seed), OrderedDict()
        for row in self.index["shards"]:
            if sha256(self.root / row["file"]) != row["sha256"]:
                raise ValueError("Pair shard hash mismatch")
        self.draws = 0

    def state_dict(self):
        return {"sampling_rng": self.rng.getstate(), "draws": self.draws}

    def load_state_dict(self, state):
        self.rng.setstate(state["sampling_rng"])
        self.draws = state["draws"]

    def batch(self, count, device):
        # One size-weighted shard per microbatch, then uniform rows within it.
        # Every pair retains equal marginal probability without loading a 6MB
        # shard for almost every single image. Last-shard weighting is exact.
        shard = self.rng.randrange(self.index["count"]) // self.index["shard_size"]
        if shard not in self.cache:
            self.cache[shard] = torch.load(self.root / self.index["shards"][shard]["file"],
                                          map_location="cpu", weights_only=True)
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        value = self.cache[shard]
        offsets = torch.tensor([self.rng.randrange(len(value["category"])) for _ in range(count)])
        self.draws += count
        return tuple(value[key][offsets].to(device=device, dtype=torch.long if key == "category" else torch.float32)
                     for key in ("z", "x", "category"))
