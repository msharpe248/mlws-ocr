"""The engine profiles under configs/ keep their promises.

classic is the reference engine and default.toml is classic; pure is classic
with both networks switched off and nothing else changed; neural differs
from classic only in recognize/decode parameters (a network enters as a
gated term, never as a different stage list).  Every parameter a profile
sets must be one its stage declares, so a typo cannot silently run the
default.
"""
from pathlib import Path

import pytest

import mlws_ocr.cleanup, mlws_ocr.layout  # noqa: F401
import mlws_ocr.glyph.components, mlws_ocr.recognize.stage  # noqa: F401
import mlws_ocr.decode, mlws_ocr.adapt  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
NET_SWITCHES = {("recognize", "mlp_path"): "", ("decode", "char_lm"): ""}


def _specs(name):
    return {(i, s.slot): s for i, s in enumerate(load_config(CONFIGS / name).stages)}


def _stage_list(name):
    return [(s.slot, s.impl) for s in load_config(CONFIGS / name).stages]


def test_default_is_classic():
    assert _specs("default.toml").keys() == _specs("classic.toml").keys()
    for key, spec in _specs("default.toml").items():
        other = _specs("classic.toml")[key]
        assert (spec.impl, spec.params) == (other.impl, other.params), key


def test_pure_is_classic_without_networks():
    assert _stage_list("pure.toml") == _stage_list("classic.toml")
    pure, classic = _specs("pure.toml"), _specs("classic.toml")
    for key, spec in pure.items():
        extra = {k: v for k, v in spec.params.items()
                 if classic[key].params.get(k) != v}
        expected = {k: v for (slot, k), v in NET_SWITCHES.items() if slot == key[1]}
        assert extra == expected, key


def test_neural_shares_the_stage_list_with_classic():
    """Same slots in the same order; the neural profile may swap ONLY the
    decoder's implementation, and then only for a subclass of the classic
    beam decoder (the line reader, decode/lineread.py), so every classic
    term is still there beneath the network's."""
    from mlws_ocr.core import registry
    from mlws_ocr.decode.beam import BeamDecode
    neural_list, classic_list = _stage_list("neural.toml"), _stage_list("classic.toml")
    assert [s for s, _ in neural_list] == [s for s, _ in classic_list]
    for (slot, impl_n), (_, impl_c) in zip(neural_list, classic_list):
        if impl_n != impl_c:
            assert slot == "decode", slot
            assert issubclass(registry.get(slot, impl_n), BeamDecode), impl_n
    neural, classic = _specs("neural.toml"), _specs("classic.toml")
    for key, spec in neural.items():
        if spec.params != classic.get(key, classic.get(("decode", "beam"))).params:
            assert key[1] in ("recognize", "decode"), key


@pytest.mark.parametrize("name", ["classic.toml", "pure.toml", "neural.toml"])
def test_profile_parameters_are_declared(name):
    for spec in load_config(CONFIGS / name).stages:
        declared = registry.get(spec.slot, spec.impl).defaults
        unknown = set(spec.params) - set(declared)
        assert not unknown, f"{name} [stage.{spec.slot}] sets unknown {unknown}"
