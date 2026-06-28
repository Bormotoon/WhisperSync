"""Tests for atempo chain decomposition."""

from __future__ import annotations

import math

from bormosync.engine.media import build_atempo_chain


def _eval_chain(chain: list[str]) -> float:
    product = 1.0
    for f in chain:
        val = float(f.replace("atempo=", ""))
        product *= val
    return product


def test_atempo_chain_identity() -> None:
    chain = build_atempo_chain(1.0)
    assert len(chain) == 1
    product = _eval_chain(chain)
    assert abs(product - 1.0) < 1e-4


def test_atempo_chain_normal_range() -> None:
    for factor in [0.5, 0.75, 1.0, 1.5, 2.0]:
        chain = build_atempo_chain(factor)
        product = _eval_chain(chain)
        assert abs(product - factor) < 1e-4, f"factor={factor}, product={product}"
        for f in chain:
            val = float(f.replace("atempo=", ""))
            assert 0.5 <= val <= 2.0, f"filter {f} out of range"


def test_atempo_chain_extreme_slow() -> None:
    factor = 0.1
    chain = build_atempo_chain(factor)
    product = _eval_chain(chain)
    assert abs(product - factor) < 1e-3
    for f in chain:
        val = float(f.replace("atempo=", ""))
        assert 0.5 <= val <= 2.0


def test_atempo_chain_extreme_fast() -> None:
    factor = 8.0
    chain = build_atempo_chain(factor)
    product = _eval_chain(chain)
    assert abs(product - factor) < 1e-2
    for f in chain:
        val = float(f.replace("atempo=", ""))
        assert 0.5 <= val <= 2.0


def test_atempo_chain_typical_drift() -> None:
    for k in [0.9995, 0.999, 1.001, 1.005]:
        factor = 1.0 / k
        chain = build_atempo_chain(factor)
        product = _eval_chain(chain)
        assert abs(product - factor) < 1e-4
