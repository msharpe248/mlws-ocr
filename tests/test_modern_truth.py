import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_modern_set import clean_truth  # noqa: E402


def test_slug_lines_are_dropped_and_glued_line_numbers_split():
    raw = "\n".join([
        "Section 161 of the Energy Policy and Conservation 5",
        "Act (42 U.S.C. 6241) is amended by adding at the end 6",
        "VerDate Sep<11>2014 03:20 Jan 14, 2023 Jkt 039200 PO 00000 Frm 00002 Fmt 6652 Sfmt 6201 E:\\BILLS\\H21.IH H21",
        "vere energy supply disruption, the jurisdic-17",
        "1993 Annual Report 12",        # a number that is content stays glued? no: split too
    ])
    out = clean_truth(raw)
    assert "VerDate" not in out and "Sfmt" not in out and "E:\\" not in out
    assert "jurisdic- 17" in out          # line number un-glued from the fragment
    assert "Conservation 5" in out        # already-separate numbers untouched
    assert "1993" in out
