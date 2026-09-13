"""Photos must leave the ink; text must stay."""
import numpy as np

import mlws_ocr.layout  # noqa: F401
from mlws_ocr.core import registry
from mlws_ocr.core.artifacts import Page
from mlws_ocr.factory.synth import render_text_page


def test_photo_removed_text_kept(font_path):
    img = render_text_page(["the quick brown fox jumps over the dog"] * 6,
                           font_path, px_height=32)
    # Paste a solid "photo" blob to the right of the text.
    h, w = img.shape
    img = np.concatenate([img, np.ones((h, 400), np.float32)], axis=1)
    img[40:h - 40, w + 30:w + 370] = 0.05
    binary = img < 0.5
    text_ink = binary.copy()
    text_ink[:, w:] = False

    page = Page(gray=img, binary=binary, dpi=300.0)
    stage = registry.get("imagezones", "density")()
    out, dbg = stage.run(page)

    assert dbg.scalars["n_zones"] >= 1
    assert not out.binary[:, w + 40:].any(), "photo ink survived"
    kept = (out.binary & text_ink).sum() / text_ink.sum()
    assert kept > 0.98, f"text ink lost: kept only {kept:.1%}"


def test_line_art_removed_text_kept(font_path):
    """Hollow line art (an illustration outline) is a graphic too.

    Regression fixture for UNLV 8509: a mailbag drawing's dense half was
    zoned while its line-art envelope spill stayed in the ink, welded two
    paragraphs into one unsplittable "line", and 348 chars vanished.
    """
    img = render_text_page(["the quick brown fox jumps over the dog"] * 8,
                           font_path, px_height=32)
    h, w = img.shape
    # A drawing about 1.2 x 2.8 inches at 300 dpi (the UNLV mailbag was
    # of that order), beside the text with a margin.
    img = np.concatenate([np.ones((max(h, 1000), 1200), np.float32),
                          np.pad(img, ((0, max(1000 - h, 0)), (0, 0)), constant_values=1.0)], axis=1)
    # Dense "illustration core": a filled blob at the top left...
    img[60:420, 180:1020] = 0.05
    # ...shedding hollow line-art below it (nested open rectangles two
    # strokes thick, large in both dimensions but sparsely filled).
    for y0, x0, y1, x1 in [(450, 180, 900, 990), (480, 210, 870, 960),
                           (510, 240, 840, 930)]:
        img[y0:y1, x0:x0 + 4] = 0.05
        img[y0:y1, x1 - 4:x1] = 0.05
        img[y0:y0 + 4, x0:x1] = 0.05
    # The stage's size fractions are fractions of a LETTER-SIZE page (a
    # block handed in alone must not see its own letters as art), so the
    # fixture sits on a page-size canvas as the UNLV original did.
    canvas = np.ones((3300, 2550), np.float32)
    canvas[: img.shape[0], : img.shape[1]] = img
    img = canvas
    binary = img < 0.5
    text_ink = binary.copy()
    text_ink[:, :1200] = False

    page = Page(gray=img, binary=binary, dpi=300.0)
    out, dbg = registry.get("imagezones", "density")().run(page)

    art = out.binary[:, :1170]
    # (the outermost rectangle's far edge is the fixture's own margin case;
    # the drawing as a whole must go)
    assert art.sum() < 0.08 * binary[:, :1170].sum(), "line art survived"
    kept = (out.binary & text_ink).sum() / text_ink.sum()
    assert kept > 0.98, f"text ink lost: kept only {kept:.1%}"


def test_crumbs_inside_an_emblem_are_art(font_path):
    """An emblem's glyph-sized crumbs sit inside its box; they are art, not
    lone letters (census page 8519 shed 'x', 'u', 'J' from a church logo)."""
    img = render_text_page(["the quick brown fox jumps over the dog"] * 6,
                           font_path, px_height=32)
    h, w = img.shape
    img = np.concatenate([img, np.ones((h, 400), np.float32)], axis=1)
    img[40:h - 40, w + 30:w + 370] = 0.05              # the emblem: a solid block
    img[100:h - 100, w + 90:w + 310] = 1.0             # ...hollowed out
    rng = np.random.default_rng(1)
    for _ in range(12):                                # glyph-sized crumbs inside it
        y, x = rng.integers(110, h - 130), rng.integers(100, 290)
        img[y:y + 14, w + x:w + x + 9] = 0.05
    binary = img < 0.5
    text_ink = binary.copy(); text_ink[:, w:] = False

    out, dbg = registry.get("imagezones", "density")().run(Page(gray=img, binary=binary, dpi=300.0))
    assert dbg.scalars["n_zones"] >= 1
    assert not out.binary[:, w + 40:].any(), "emblem crumbs survived as text"
    kept = (out.binary & text_ink).sum() / text_ink.sum()
    assert kept > 0.98, f"text ink lost: kept only {kept:.1%}"
