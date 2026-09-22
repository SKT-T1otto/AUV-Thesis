"""Strict B2 checkpoint loading for evaluation."""
from ..common.checkpoint import load_checkpoint as _load_checkpoint, load_model


def load_checkpoint(path):
    return _load_checkpoint(path, expected_baseline="B2_direct_mc")
