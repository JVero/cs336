from pathlib import Path
import pathlib 
from collections import defaultdict
import os

def build_occurrence(paths: list[Path]) -> dict[int, int]:
    occurrences = defaultdict(int)
    for if_path in paths:
        with open(if_path, "r") as f:
            for line in f:
                text_hash = hash(line)
                occurrences[text_hash] += 1
    return occurrences

def dedup(paths: list[Path | os.PathLike], out_dir: pathlib.Path | os.PathLike):
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise ValueError(f"{out_dir=} is not a directory")
    ipaths: list[Path] = [Path(path) for path in paths]
    occurrences = build_occurrence(ipaths)
    for if_path in ipaths:
        of_path = out_dir / if_path.name
        with open(if_path, "r") as fi, open(of_path, "w") as fo:
            for line in fi:
                if occurrences[hash(line)] == 1:
                    fo.write(line)
                