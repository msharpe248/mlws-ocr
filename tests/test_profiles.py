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
# the word corrector is on in classic and off in pure: its image check is a network (2026-09-25)
NET_SWITCHES = {("recognize", "mlp_path"): "", ("decode", "char_lm"): "", ("correct", "enabled"): False}


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
            if slot == "blocks":
                # the segmenter judge (owner's decision, 2026-09-26): newspapers and
                # magazines only; every other page keeps classic's XY-cut unchanged
                assert impl_n == "judged" and impl_c == "xycut", (impl_n, impl_c)
                continue
            assert slot == "decode", slot
            assert issubclass(registry.get(slot, impl_n), BeamDecode), impl_n
    neural, classic = _specs("neural.toml"), _specs("classic.toml")
    # table structure, output only (owner's decision 2026-09-28): the tables the page
    # reports may differ; nothing that changes what the reader sees
    # display equations kept from the reader and pictures cut out (owner's decision 2026-10-04,
    # v0.18.5), and read (2026-10-06; line by line and confirming unnumbered displays, 2026-10-12):
    # the neural profiles only; the classic reference reads as before; and the newspaper line re-split
    # (owner's decision 2026-10-13, v0.18.16); and statements read by their figures (v0.18.20);
    # and 41 of neural-table's table settings with the detector's prose gate (owner's decision 2026-10-20, v0.18.23);
    # and a whitespace table's header from its words, its section rows whole (owner's decision 2026-10-21, v0.18.24)
    # and a nested row group's labels kept in their columns (owner's decision 2026-10-22, v0.18.25)
    # and a sparse stub column kept (owner's decision 2026-10-25, v0.18.26)
    table_reading = {"cell_marks", "cell_order_by_line", "dollar_s", "join_digit_groups", "nest_side_by_side", "span_labels", "span_labels_nested", "table_stub_columns",
                     "table_cell_conf_path", "table_cell_lines", "table_dash_aspect", "table_dashes", "table_det_max_prose", "table_det_mode", "table_det_path",
                     "table_group_rowspans", "table_header_from_words", "table_section_rows", "table_heading_rows", "table_item_rows", "table_label_rowspans", "table_net_figure_rows",
                     "table_net_path", "table_net_row_join", "table_paren_columns", "table_rebuild_centred", "table_rebuild_header",
                     "table_seq_mode", "table_seq_path", "table_seq_select", "table_split_keep_rows", "table_split_path",
                     "table_split_select", "table_wordrel_mode", "table_wordrel_path", "table_wordrel_select", "trim_notes",
                     "trim_notes_rows", "ws_detector", "ws_mesh_min_spines", "ws_table_thin_grids"}
    table_only = {"tables": {"spans", "open_sides", "nested", "diagonals", "min_row_ink"},
                  "rulings": {"edge_bars", "faint_depth"},
                  "lines": {"equations", "equations_beside", "resplit_glyph_ref", "stacked_chunks"},
                  "output": {"split_words_at_cells", "fix_figure_columns", "check_arithmetic", "ws_detect",
                             "pictures", "math_reader_path", "math_beam", "math_split_lines", "math_display_logp", "math_split_at_numbers",
                             "align_figures"} | table_reading}
    for key, spec in neural.items():
        c = classic.get(key, classic.get(("decode", "beam"))).params
        if key[1] in table_only:
            diff = {k for k in set(spec.params) | set(c) if spec.params.get(k) != c.get(k)}
            assert diff <= table_only[key[1]], (key, diff - table_only[key[1]])
            continue
        if spec.params != c:
            # the word corrector: on in classic, an option (off) in neural -- the owner's
            # decision of 2026-09-25; the neural profile carries its own confusion table
            # small type enlarged and declared 300 dpi: neural's alone (owner's decision 2026-10-18, v0.18.21)
            # quarter and half turns found by reading need neural's line reader (owner's decision 2026-10-22, v0.18.25)
            if key[1] == "deskew":
                assert {k for k in set(spec.params) | set(c) if spec.params.get(k) != c.get(k)} == {"turns"}, key
                continue
            assert key[1] in ("recognize", "decode", "correct", "blocks", "magnify"), key


@pytest.mark.parametrize("name", ["classic.toml", "pure.toml", "neural.toml", "neural-table.toml"])
def test_profile_parameters_are_declared(name):
    for spec in load_config(CONFIGS / name).stages:
        declared = registry.get(spec.slot, spec.impl).defaults
        unknown = set(spec.params) - set(declared)
        assert not unknown, f"{name} [stage.{spec.slot}] sets unknown {unknown}"


def test_neural_profile_keeps_its_decoder_identity():
    """The line reader's mode, rule and paths are the neural profile's identity; an edit
    that drops one of them silently changes every measurement (2026-09-21: a config
    rewrite lost line_mode and line_choose_rule and six rows were measured wrong)."""
    p = {s.slot: s for s in load_config(CONFIGS / "neural.toml").stages}["decode"].params
    assert p["line_mode"] == "choose" and p["line_choose_rule"] == "calibrated"
    assert p["line_model_path"] == "data/seq_line_csl_en.npz+data/seq_line_csl_en_2.npz+data/seq_line_csl_en_3.npz" and p["line_choice_path"] == "data/linechoice_csl.npz"
    # the grey ensemble must be read from grey strips: read from binary ones it silently loses
    assert p["line_source"] == "gray"
    assert p["seq_path"] == "data/seq_en.npz"


def test_command_line_sets_reach_the_stage():
    from mlws_ocr.core.config import apply_sets, parse_sets
    sets = parse_sets(["correct.enabled=true", "despeckle.min_area_300dpi=8", 'correct.confusions_path="x.json"'])
    assert sets == {"correct": {"enabled": True, "confusions_path": "x.json"}, "despeckle": {"min_area_300dpi": 8}}
    cfg = apply_sets(load_config(CONFIGS / "neural.toml"), sets)
    spec = next(s for s in cfg.stages if s.slot == "correct")
    assert spec.params["enabled"] is True and spec.params["confusions_path"] == "x.json"
    with pytest.raises(ValueError):
        apply_sets(load_config(CONFIGS / "neural.toml"), parse_sets(["nosuchslot.x=1"]))


def test_neural_table_is_neural_plus_the_table_options():
    """The table profile may differ from neural ONLY in the table options that
    change what the reader sees (and its own reader with that reader's judge); everything else stays neural's.
    The report-page options of v0.18.20 (blocks.gutter_sections, output.align_figures) and v0.18.21's
    small-type magnify and v0.18.22's spread cut are neural's alone: never measured on the table sets.
    Since v0.18.23 neural carries 41 of these table settings too, and the detector's prose gate
    (output.table_det_max_prose) neural-table does not, and since v0.18.24 the header from words and
    section rows (output.table_header_from_words, output.table_section_rows), and since v0.18.25 the nested row
    labels (output.span_labels_nested) and quarter turns (deskew.turns, not trialled on the table profile), and since
    v0.18.26 the sparse stub column (output.table_stub_columns) and I/l by the neighbouring words (correct.context_repair); those left out of neural (blocks.keep_dashes,
    imagezones.keep_grids, lines.in_cells, rulings.short_in_grid_300dpi, components.dotted_rule_text,
    tables.broken_rules, the decode settings) cost text on forms, receipts, magazines and ordinary pages."""
    allowed = {"magnify": {"max_scale", "target_px", "max_page_px", "type_sets_dpi"}, "deskew": {"edge_nearest", "ink_fallback", "turns"}, "illumination": {"stretch_low", "invert_dark", "grey_pictures", "invert_rect"}, "imagezones": {"keep_grids"}, "components": {"dotted_rule_text"}, "rulings": {"short_in_grid_300dpi", "edge_bars", "faint_depth"},
               "tables": {"diagonals", "min_row_ink", "broken_rules"}, "decode": {"line_gray_rules_out", "figure_cells", "rotated_cells", "line_model_path", "line_choice_path", "line_gap_split", "line_model_path_table", "line_model_path_symbols", "line_dash_measure", "line_symbol_net"}, "lines": {"in_cells", "stacked_chunks"}, "blocks": {"keep_dashes", "gutter_sections", "gutter_spread"},
               "correct": {"context_repair"},
               "output": {"table_net_path", "table_net_row_join", "table_net_figure_rows", "ws_detector", "ws_mesh_min_spines", "cell_order_by_line",
                          "table_det_path", "table_det_mode", "table_det_max_prose", "table_header_from_words", "table_section_rows", "table_split_path", "table_split_select", "table_split_keep_rows", "table_dashes", "ws_table_thin_grids", "span_labels", "span_labels_nested", "table_stub_columns",
                          "nest_side_by_side", "pictures", "table_dash_aspect", "table_group_rowspans", "table_rebuild_header", "table_heading_rows", "table_rebuild_centred", "table_cell_conf_path", "join_digit_groups", "dollar_s", "trim_notes", "cell_marks", "trim_notes_rows", "table_cell_lines", "table_label_rowspans", "table_wordrel_path", "table_wordrel_mode", "table_wordrel_select", "table_seq_path", "table_seq_mode", "table_seq_select", "table_item_rows", "table_paren_columns", "align_figures"}}
    a, b = _specs("neural-table.toml"), _specs("neural.toml")
    assert list(a) == list(b)
    for key in a:
        diff = {k for k in set(a[key].params) | set(b[key].params) if a[key].params.get(k) != b[key].params.get(k)}
        assert diff <= allowed.get(key[1], set()), (key, diff)
