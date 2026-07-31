"""Registry of FOLDABLE objects for the Fold operator (real2sim2real extension).

A foldable garment starts as its FLAT collision geometry (thin slab at the perceived
pose). The Fold operator swaps its collision spheres to the FOLDED bundle set (from the
Newton twin's measured fold curve) for every timestep at and after the fold. Populated
by the environment builder before run_cutamp; keyed by object name.

Entry: {"folded_spheres": Tensor[n, 4] in the OBJECT frame — Tetris convention: the top
grasp-handle sphere at z = 0, body hanging below (see run_sam3d_pack gotcha #2)}.
"""
from typing import Dict

FOLDABLE: Dict[str, dict] = {}


def register_foldable(name: str, folded_spheres) -> None:
    FOLDABLE[name] = {"folded_spheres": folded_spheres}


def clear() -> None:
    FOLDABLE.clear()
