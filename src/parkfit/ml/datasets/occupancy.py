"""CNRPark-EXT: is this particular parking space occupied right now?

This is the question the product actually asks, and getting here took an embarrassing
detour. The detector spent weeks trying to *find* vehicles in a street scene, which is
the hard version of the problem. Amsterdam already publishes 210,247 parking bays as
surveyed polygons, so the location of every bay is known before a single pixel is read.
What is not known is whether each one currently has a car in it, and that is a per-crop
binary question rather than a detection problem.

CNRPark-EXT is exactly that question with 144,965 labelled answers: 150x150 crops of real
parking spaces from nine fixed cameras over about three months, each marked busy or free,
under sunny, overcast and rainy skies. The geometry matches the deployment too. These are
fixed cameras looking at spaces whose corners somebody surveyed once, which is the same
arrangement as a street camera pointed at a kerb whose polygon came out of the municipal
register.

**Three ways to split it, and they answer different questions.** The authors ship an
official split by *day*, which is the published benchmark and the number comparable to
other people's work. Splitting by *camera* asks whether the model transfers to a viewpoint
it has never seen, which is the harder question and the one that matters for pointing this
at Amsterdam. Splitting by *weather* asks whether it learned cars or learned sunshine. All
three are supported and the trainer reports whichever it was given.

Patches are read from disk rather than held in memory. At 128x128 the full set is 7.1 GB
as uint8, which does not fit beside anything else on this machine, and the files are 3 KB
JPEGs that decode fast enough to keep a GPU fed with a few worker processes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Where `pf occupancy fetch` puts things.
DEFAULT_ROOT = Path("data/parking_ds")

#: The label column is 1 for busy and 0 for free, which is the order the head predicts.
CLASS_NAMES = ("free", "occupied")

#: CNRPark-EXT crops are 150x150. 96 keeps a parked car comfortably readable while
#: letting a batch of 256 sit in VRAM next to everything else.
PATCH_SIZE = 96


@dataclass(frozen=True)
class Patch:
    """One labelled parking space crop."""

    path: Path
    label: int
    camera: str
    weather: str
    day: str

    @staticmethod
    def from_line(line: str, patches_root: Path) -> Patch | None:
        """Parse a line of the shipped split files: ``WEATHER/DAY/cameraN/file.jpg 1``."""
        parts = line.strip().split()
        if len(parts) != 2:
            return None
        rel, raw = parts
        try:
            label = int(raw)
        except ValueError:
            return None
        pieces = rel.split("/")
        if len(pieces) < 4:
            return None
        weather, day, camera = pieces[0], pieces[1], pieces[2]
        return Patch(
            path=patches_root / rel,
            label=label,
            camera=camera,
            weather=weather.lower(),
            day=day,
        )


def read_split(split_file: Path, patches_root: Path) -> list[Patch]:
    """Read one of the shipped split files into patches, skipping anything missing."""
    if not split_file.exists():
        raise FileNotFoundError(f"no split file at {split_file}")

    patches: list[Patch] = []
    missing = 0
    for line in split_file.read_text(encoding="utf-8").splitlines():
        patch = Patch.from_line(line, patches_root)
        if patch is None:
            continue
        if not patch.path.exists():
            missing += 1
            continue
        patches.append(patch)

    if missing:
        log.warning("%s: %d listed patches are not on disk", split_file.name, missing)
    return patches


def split_by_camera(patches: list[Patch], holdout: set[str]) -> tuple[list[Patch], list[Patch]]:
    """Train and test on disjoint cameras.

    Harder than the official day split and closer to the real deployment question: the
    Amsterdam cameras are not any of these nine, so what matters is whether the thing
    transfers to a viewpoint it has never seen rather than to a Tuesday it has never seen.
    """
    train = [p for p in patches if p.camera not in holdout]
    test = [p for p in patches if p.camera in holdout]
    return train, test


def split_by_weather(patches: list[Patch], holdout: set[str]) -> tuple[list[Patch], list[Patch]]:
    """Train and test on disjoint weather. Answers whether it learned cars or sunshine."""
    holdout = {w.lower() for w in holdout}
    train = [p for p in patches if p.weather not in holdout]
    test = [p for p in patches if p.weather in holdout]
    return train, test


def describe(patches: list[Patch]) -> dict:
    """Counts a caller can print or write into a report."""
    from collections import Counter

    return {
        "patches": len(patches),
        "occupied": sum(1 for p in patches if p.label == 1),
        "free": sum(1 for p in patches if p.label == 0),
        "cameras": dict(sorted(Counter(p.camera for p in patches).items())),
        "weather": dict(sorted(Counter(p.weather for p in patches).items())),
        "days": len({p.day for p in patches}),
    }


class PatchSet:
    """A map-style dataset over parking-space crops.

    Deliberately a plain module-level class rather than one defined inside a factory.
    DataLoader workers on Windows start by spawning and pickling the dataset, and a class
    defined in a function body cannot be pickled: the run dies with "Can't get local
    object". torch's DataLoader only needs ``__len__`` and ``__getitem__``, so there is
    nothing to inherit and nothing lost by keeping it importable.

    Augmentation is mild and mostly photometric. These are fixed cameras, so a space is
    always seen from the same angle and rotating it would teach variation the deployment
    never has. What actually changes between one frame and the next is the sun moving and
    the weather turning, so that is what gets simulated.
    """

    def __init__(self, patches: list[Patch], *, train: bool, size: int = PATCH_SIZE) -> None:
        self.patches = patches
        self.train = train
        self.size = size

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image

        patch = self.patches[index]
        with Image.open(patch.path) as handle:
            image = handle.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
            array = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        array = array.view(self.size, self.size, 3).permute(2, 0, 1).float().div_(255.0)

        if self.train:
            if torch.rand(()) < 0.5:
                array = torch.flip(array, dims=[2])
            if torch.rand(()) < 0.7:
                brightness = 0.7 + 0.6 * float(torch.rand(()))
                contrast = 0.8 + 0.4 * float(torch.rand(()))
                mean = array.mean()
                array = ((array * brightness - mean) * contrast + mean).clamp_(0.0, 1.0)

        return array, patch.label


def build_dataset(patches: list[Patch], *, train: bool, size: int = PATCH_SIZE) -> PatchSet:
    """A torch-compatible dataset over the crops."""
    return PatchSet(patches, train=train, size=size)
