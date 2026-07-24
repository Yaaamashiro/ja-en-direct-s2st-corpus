from __future__ import annotations

import hashlib
import random

import numpy as np


def deterministic_seed(pair_id: str, language: str, attempt: int) -> int:
    material = f"{pair_id}\0{language}\0{attempt}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return value & 0x7FFF_FFFF_FFFF_FFFF


def seed_everything(seed: int, torch_module: object) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
