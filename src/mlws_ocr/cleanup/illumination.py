"""Illumination correction: flatten uneven page lighting.

Scanners and photocopiers leave low-frequency brightness fields (dark
corners, gradients along the platen).  Estimate the background by heavily
median-filtering a downsampled copy (text strokes are far smaller than the
filter window, so only paper survives), then divide the image by it.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core.artifacts import Page
from ..core.registry import register
from ..core.stage import DebugBundle, Stage


@register
class MedianBackgroundIllumination(Stage):
    slot = "illumination"
    impl = "median_background"
    defaults = {
        "downsample": 8,     # estimate background on a 1/8 scale copy
        "window": 31,        # median window at the downsampled scale
        "floor": 0.05,       # lower clamp on background to avoid blowups
        "frame_dark": 0.0,   # a scanner's black frame: where the RAW gray's local mean (over
                             # frame_blur_300dpi px) is below this and the dark region touches
                             # the image edge, the corrected image is set to paper.  0 = off.
                             # Left in, the median background inside a 75-px frame IS the
                             # frame, division turns it paper-white with speckle, and Sauvola
                             # made 176 lines of a 15-line Library of Congress page (2026-09-22).
        "stretch_low": 0.0,  # > 0: a page whose darkest 1% is still lighter than this after the
                             # correction (a dim photo of faint print: ink at 0.8 of the paper)
                             # is stretched so that level is black -- Sauvola found 0.4% of a
                             # CORD receipt's page as ink and no line (2026-09-29).  0 = off
        "stretch_max_gain": 5.0,
        "invert_mid": 0.0,     # > 0: in a dark region only the extremes are inverted -- below this and
                               # above 1 - this -- the middle tones kept (lightness flipped where it is
                               # ground or letters, left where it is an image, an icon, a grey fill;
                               # the owner's idea, 2026-10-03); 0 = the whole region inverted
        "invert_dark": False,  # light text on a dark ground -- a screenshot in dark mode, a coloured
                               # header band with white text, a black title bar -- turned to dark on
                               # light before anything else (dark_ground), as print never needs; also
                               # keeps a dark page from being taken for a scanner frame (2026-10-03)
        "invert_text": False,  # light text LINES on any dark ground inverted, line by line
                               # (light_text_lines): reverse video on a report's black page or a
                               # photograph's white caption, where invert_dark takes only whole
                               # rectangular grounds and a dark photograph with them (2026-10-13)
        "invert_text_page": 0.45,  # ...only on a page at least this share dark ground (a reverse-video
                                   # page: report pages that gained 0.50-0.96, pages that lost 0.41 and
                                   # under, where a box inverted mid-page merged the columns round it)
        "grey_pictures": False,  # pictures found on the GREY page first (grey_pictures): a photograph's
                                 # varied mid-tones, which binarization scatters (a mid-grey photo comes
                                 # out nearly white, its edge a dotted frame) and the dark-ground test
                                 # takes for a band (a dark photo inverted to nothing).  They are kept
                                 # out of the inversion and handed to imagezones as picture zones
                                 # (meta["picture_boxes"]; 2026-10-03)
        "invert_rect": 0.85,     # a dark region is a ground only if it fills this much of its box; 0.95
                                 # keeps a logo's bold letters (0.85 after deskew on the payroll forms)
                                 # from being inverted, where a band or a dark page fills 0.98-1.00
                                 # (measured on the screenshot set's dark and coloured-header tables,
                                 # 2026-10-03)
        "frame_blur_300dpi": 15,
        "frame_min_edges": 3,  # a frame touches at least three of the image's four edges
                               # (a surround touches four, a lid's strip three); a dark
                               # photograph bleeding off a corner touches two and stays.
                               # Hollowness and thinness were tried first: a halftone's
                               # patchy dark regions pass a hollowness test and cost a
                               # newspaper page 4 words (2026-09-22).
    }

    def run(self, page: Page) -> tuple[Page, DebugBundle]:
        p = self.params
        gray = page.gray
        inverted = 0
        pictures, pic_boxes = None, []
        # on a page magnified 2x or more (a 72-dpi crop) type itself is mid-grey blur, and the
        # tone cue cannot part it from a photograph: a whole PubTables-1M table was taken
        # for one picture and read empty (2026-10-03)
        if p["grey_pictures"] and float(page.meta.get("magnify_scale") or 1.0) < 2.0:
            pictures, pic_boxes = grey_pictures(gray, page.dpi or 300.0)
        if p["invert_dark"]:
            region = dark_ground(gray, page.dpi or 300.0, rect=float(p["invert_rect"]))
            if pictures is not None:
                region &= ~pictures
            inverted = int(region.sum())
            if inverted:
                gray = invert_regions(gray, region, float(p["invert_mid"]))
        text_lines = 0
        if p["invert_text"]:
            region, text_lines = light_text_lines(gray, page.dpi or 300.0,
                                                  page_share=float(p["invert_text_page"]))
            if pictures is not None:
                region &= ~pictures
            if text_lines:
                inverted += int(region.sum())
                gray = invert_regions(gray, region, float(p["invert_mid"]))

        small = ndimage.zoom(gray, 1.0 / p["downsample"], order=1)
        bg_small = ndimage.median_filter(small, size=p["window"], mode="nearest")
        background = ndimage.zoom(bg_small, np.array(gray.shape) / np.array(bg_small.shape),
                                  order=1)
        background = np.clip(background[:gray.shape[0], :gray.shape[1]], p["floor"], None)

        corrected = np.clip(gray / background, 0.0, 1.0).astype(np.float32)
        frame_px = 0
        if p["frame_dark"] > 0:
            size = max(3, round(p["frame_blur_300dpi"] * page.dpi / 300.0))
            frame = scanner_frame(gray, float(p["frame_dark"]), size, int(p["frame_min_edges"]))
            frame_px = int(frame.sum())
            if frame_px:
                corrected[frame] = 1.0

        stretched = 0.0
        if p["stretch_low"] > 0:
            lo = float(np.percentile(corrected, 1))
            if lo > p["stretch_low"]:
                gain = min(float(p["stretch_max_gain"]), 1.0 / max(1e-3, 1.0 - lo))
                corrected = np.clip(1.0 - (1.0 - corrected) * gain, 0.0, 1.0).astype(np.float32)
                stretched = round(gain, 2)

        out = page.evolve(gray=corrected)
        out.meta.setdefault("corrections", {})["illumination"] = "median_background"
        if p["grey_pictures"]:
            out.meta["picture_boxes"] = pic_boxes
        debug = DebugBundle(
            images={"input": gray, "background": background, "corrected": corrected},
            scalars={"background_min": round(float(background.min()), 3),
                     "background_max": round(float(background.max()), 3),
                     "frame_pixels": frame_px, "stretch_gain": stretched,
                     "inverted_share": round(inverted / max(1, gray.size), 3),
                     "inverted_text_lines": text_lines,
                     "grey_pictures": len(pic_boxes)},
        )
        return out, debug


def invert_regions(gray: np.ndarray, region: np.ndarray, mid: float = 0.0) -> np.ndarray:
    """Each connected dark-ground region inverted and scaled so that its own
    ground becomes paper (a mid-grey header band inverts to light grey, not
    white, and its edge then reads as a rule), and the anti-aliased ring of
    edge pixels just outside it set to paper too."""
    out = gray.astype(np.float32).copy()
    lab, n = ndimage.label(region)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == i
        ground = float(np.median(gray[sl][m]))          # the band's own grey (its letters are few)
        v = gray[sl][m]
        inv = np.clip((1.0 - v) / max(1e-3, 1.0 - ground), 0.0, 1.0)
        if mid > 0:                                      # only the extremes: ground and letters
            inv = np.where((v < mid) | (v > 1.0 - mid), inv, v)
        out[sl][m] = inv
    ring = ndimage.binary_dilation(region, iterations=2) & ~region & (gray < 0.95)
    out[ring] = 1.0
    return out


def grey_pictures(gray: np.ndarray, dpi: float, lo: float = 0.12, hi: float = 0.88,
                  share: float = 0.5, flat: float = 0.5, min_in: float = 0.4, max_page: float = 0.6,
                  max_paper: float = 0.5):
    """PICTURES on the grey page -- photographs, shaded artwork -- as (mask, boxes).

    Print is two-toned: paper and ink, mid-tones only on a stroke's
    anti-aliased edge, so over a window about a text line tall (40 px at
    300 dpi) at most a small share of a text region is mid-grey.  A
    photograph is mostly mid-tones.  Where at least ``share`` of the window
    is between ``lo`` and ``hi`` is a picture's core, grown through the
    non-paper pixels it touches; a region at least ``min_in`` inch on both
    sides, under ``max_page`` of the page and at most ``max_paper`` paper
    inside its box (blurred small type is mid-grey strokes on paper), is
    kept when it is VARIED -- under ``flat`` of its non-paper pixels
    within 0.03 of their median -- because a coloured header band or a grey
    fill is mid-grey too, but one grey with letters on it (they are the
    dark-ground test's).  The tone-texture rule of the classic
    text / halftone / graphics block classifiers (Wahl, Wong & Casey, "Block
    segmentation and text extraction in mixed text/image documents", CGIP
    1982; Wang & Srihari, "Classification of newspaper image blocks using
    texture analysis", CVGIP 1989), here on grey levels before binarization
    rather than on the binary runs after it.  Levels are taken relative to
    the page's paper (its 90th percentile)."""
    w = max(9, int(round(40 * dpi / 300.0)) | 1)
    # levels relative to the page's own paper (a grey or dim scan's paper sits at 0.85 and its
    # noise is mid-tone: unscaled, a whole invoice page was one 'picture')
    paper = max(0.3, float(np.percentile(gray, 90)))
    gray = np.clip(gray / paper, 0.0, 1.0)
    mid = (gray > lo) & (gray < hi)
    core = ndimage.uniform_filter(mid.astype(np.float32), size=w) >= share
    keep = np.zeros(gray.shape, bool)
    boxes: list[list[int]] = []
    if not core.any():
        return keep, boxes
    grown = ndimage.binary_propagation(core, mask=gray < hi)
    lab, n = ndimage.label(ndimage.binary_closing(grown, iterations=3))
    side = min_in * dpi
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl[0].stop - sl[0].start < side or sl[1].stop - sl[1].start < side:
            continue
        if (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start) > max_page * gray.size:
            continue                       # most of the page: a shaded or dim page, not a picture on it
        if (gray[sl] >= hi).mean() > max_paper:
            continue                       # mostly paper: small blurred type (a 72-dpi crop magnified
                                           # is mid-grey strokes on paper, 0.85 of its box), not a photo (0.0)
        m = lab[sl] == i
        v = gray[sl][m & (gray[sl] < hi)]
        if v.size == 0 or np.mean(np.abs(v - np.median(v)) < 0.03) >= flat:
            continue
        keep[sl] = True                    # the whole box: a picture's light parts are picture too
        boxes.append([int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)])
    return keep, boxes


def dark_ground(gray: np.ndarray, dpi: float, dark: float = 0.45, share: float = 0.6,
                rect: float = 0.85) -> np.ndarray:
    """The pixels on a DARK GROUND, to be inverted: light text on a dark page or
    band.  Over a window about a text line tall (40 px at 300 dpi) a share of
    at least ``share`` of the pixels darker than ``dark`` is ground, not ink --
    strokes, however bold, never fill most of a line-sized window -- and the
    region is that core grown through the darkish pixels it touches (not into
    the white page round a band) with everything it encloses (the light letters
    on it); and only a region filling ``rect`` of its bounding box is a ground
    (a page, a band, a bar), where a logo's huge bold letters are ragged.  Inverse
    text detection by local polarity: the document-image rule that text is
    darker than its own surround, applied region by region (cf. Kasar, Kumar &
    Ramakrishnan, "Font and background color independent text binarization",
    CBDAR 2007, which binarizes each component against its own polarity)."""
    w = max(9, int(round(40 * dpi / 300.0)) | 1)
    frac = ndimage.uniform_filter((gray < dark).astype(np.float32), size=w)
    core = frac >= share
    if not core.any():
        return core
    # grown only through darkish pixels (the band, not the white page round it, which would
    # turn into a black frame), then closed over what it encloses (the light letters on it,
    # which a smoothed test had left out, so they came back hollow)
    grown = ndimage.binary_fill_holes(ndimage.binary_propagation(core, mask=gray < 0.6))
    # a ground is a rectangle -- a dark page, a header band, a title bar -- filling its box;
    # very large bold type (a logo's 'WHD') also fills line-sized windows but is ragged
    lab, n = ndimage.label(grown)
    keep = np.zeros_like(grown)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == i
        if m.sum() >= rect * m.size:
            keep[sl] |= m
    return keep


def light_text_lines(gray: np.ndarray, dpi: float, min_glyphs: int = 4,
                     gap: float = 1.2, solid: float = 0.8, pure: float = 0.75,
                     page_share: float = 0.0) -> tuple[np.ndarray, int]:
    """LIGHT TEXT on a dark ground, as a mask of its lines' boxes, and the
    number of lines: reverse video in a report's black page, white captions
    on a photograph, a dark banner -- found line by line, where invert_dark
    inverts whole rectangular grounds (and a dark photograph with them).

    The candidate ground is every dark region (``dark_ground`` with no
    rectangle test); inside it, a light component is a GLYPH when it has a
    letter's proportions (8-100 px tall at 300 dpi, at most six times as
    wide as tall, filling 8-90% of its box, lying wholly on the ground with
    dark round 60% of its box's border), and glyphs make a LINE when at
    least ``min_glyphs`` of them follow each other within ``gap`` letter
    heights, of similar size, on a shared baseline or top line (a
    photograph's light specks are not), AND inside one connected dark
    ground, with at least ``solid`` of the strips just above and below the
    line dark (a quarter of its height each), and at least ``pure`` of the
    light pixels in its box in its letters (a dithered photograph's specks
    chain too, among many smaller ones).  These are the polarity tests: light
    letters sit in one ground that runs round them all, where the light
    counters of a heavy black headline ('BOOK') each sit in a letter of
    their own (or, touching, in one word with white paper over it) -- the
    region's text polarity decided by its components, as
    Kasar, Kumar & Ramakrishnan decide each component's ("Font and background
    color independent text binarization", CBDAR 2007).  Each line's box,
    padded by a third of its height and clipped to the ground, is the mask
    (invert_regions inverts it), padded by 0.6 of its height along it (a
    line-end hyphen or period is too small for a glyph) and the boxes closed
    into blocks.  Light display letters cut by a photograph's highlights stay
    light: the test needs four letters in a row.  With ``page_share`` only a
    page at least that much dark ground is searched: a REVERSE-VIDEO PAGE.
    A white box or sidebar inverted mid-page joined the columns either side of
    it into lines (as a dark area it had been a picture zone between them;
    UNLV 8005_125), and an advertisement's reverse panel is often outside a
    page's truth (8023_017)."""
    s = dpi / 300.0
    ground = dark_ground(gray, dpi, rect=0.0)
    if not ground.any() or ground.mean() < page_share:
        return np.zeros_like(ground), 0
    darkpix = (gray < 0.5) & ground
    dlab, _ = ndimage.label(darkpix, structure=np.ones((3, 3), bool))
    # light components of the WHOLE page, so the paper between black letters (which the
    # grown ground also takes in) stays one huge component and never passes for a glyph
    lab, n = ndimage.label(gray >= 0.5, structure=np.ones((3, 3), bool))
    glyphs = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if not (8 * s <= h <= 100 * s and w <= 6 * h):
            continue
        comp = lab[sl] == i
        fill = float(comp.sum()) / (h * w)
        if not 0.08 <= fill <= 0.9 or not ground[sl][comp].all():
            continue
        # the dark ground round the glyph: the commonest dark label on its box's border ring
        y0, y1 = max(sl[0].start - 2, 0), min(sl[0].stop + 2, gray.shape[0])
        x0, x1 = max(sl[1].start - 2, 0), min(sl[1].stop + 2, gray.shape[1])
        ring = np.concatenate([dlab[y0, x0:x1], dlab[y1 - 1, x0:x1], dlab[y0:y1, x0], dlab[y0:y1, x1 - 1]])
        if (ring > 0).mean() < 0.6:
            continue                       # a light glyph has dark ground round it
        ring = ring[ring > 0]
        glyphs.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, int(np.bincount(ring).argmax()), i))
    if len(glyphs) < min_glyphs:
        return np.zeros_like(ground), 0
    g = np.array(glyphs)
    order = np.argsort(g[:, 0])
    g = g[order]
    parent = list(range(len(g)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a in range(len(g)):
        ha = g[a, 3] - g[a, 1]
        for b in range(a + 1, len(g)):
            if g[b, 0] - g[a, 2] > gap * ha:
                break
            hb = g[b, 3] - g[b, 1]
            if g[b, 4] != g[a, 4] or max(ha, hb) > 2 * min(ha, hb) or g[b, 0] < g[a, 0]:
                continue
            if min(g[a, 3], g[b, 3]) - max(g[a, 1], g[b, 1]) < 0.5 * min(ha, hb):
                continue
            # letters share a baseline or a top line; a photograph's light specks neither
            if min(abs(g[a, 3] - g[b, 3]), abs(g[a, 1] - g[b, 1])) > 0.2 * max(ha, hb):
                continue
            parent[find(b)] = find(a)
    groups: dict = {}
    for a in range(len(g)):
        groups.setdefault(find(a), []).append(a)
    mask = np.zeros_like(ground)
    lines, heights = 0, []
    for members in groups.values():
        if len(members) < min_glyphs:
            continue
        box = g[members]
        x0, y0, x1, y1 = box[:, 0].min(), box[:, 1].min(), box[:, 2].max(), box[:, 3].max()
        # solid ground just above and below the line: the counters of heavy black type
        # have white paper over their letters, a photograph's specks more specks
        strip = max(3, int(round(0.25 * (y1 - y0))))
        above, below = gray[max(y0 - strip, 0):y0, x0:x1], gray[y1:y1 + strip, x0:x1]
        edge = np.concatenate([above.ravel(), below.ravel()])
        if not edge.size or (edge < 0.5).mean() < solid:
            continue
        # most of the light in the line's box is its letters: a dithered photograph's
        # specks chain too, among many more specks too small to be letters
        inbox = lab[y0:y1, x0:x1]
        light_px = int((inbox > 0).sum())
        if light_px and np.isin(inbox, box[:, 5]).sum() < pure * light_px:
            continue
        pad = int(round((y1 - y0) / 3))
        padx = int(round(0.6 * (y1 - y0)))     # a line's last hyphen or period, too small for a glyph
        sl = (slice(max(y0 - pad, 0), y1 + pad), slice(max(x0 - padx, 0), x1 + padx))
        mask[sl] = True
        heights.append(y1 - y0)
        lines += 1
    if lines:
        # the lines' boxes closed into blocks (a line pitch tall, three letters wide), so the
        # ground between words, lines and a line's last letters goes too, not left as black
        # stubs to be read as ink; then clipped to the ground
        h = float(np.median(heights))
        size = (max(3, int(2 * h)), max(3, int(3 * h)))
        grown = ndimage.uniform_filter(mask.astype(np.float32), size=size) > 1e-3
        mask = (ndimage.uniform_filter(grown.astype(np.float32), size=size) > 1 - 1e-3) | mask
        # and a margin round the blocks (half a line up and down, three letters along), so a
        # short line's end inside a ragged column edge is not left a black notch
        margin = (max(3, int(h)), max(3, int(6 * h)))
        mask = (ndimage.uniform_filter(mask.astype(np.float32), size=margin) > 1e-3) & ground
    return mask, lines


def scanner_frame(gray: np.ndarray, dark: float, size: int, min_edges: int = 3) -> np.ndarray:
    """The mask of a scanner's frame.  Text is dark marks on light paper, so
    its local mean gray stays high (a text line is at most a third ink); a
    frame is a region whose local mean is dark AND that touches at least
    ``min_edges`` of the image's four edges: the black surround of a page
    or of a receipt on a dark table touches four, a scanner lid's strip
    three.  A dark photograph bleeding off a corner touches two and stays
    for the image-zone stage; so does the cut-off last line of a page.
    Candidates are the regions whose box mean of ``gray`` over ``size`` px
    is below ``dark``; the mask is grown by ``size`` to take the ragged
    transition into the paper."""
    mean = ndimage.uniform_filter(gray.astype(np.float32), size=size, mode="nearest")
    labels, n = ndimage.label(mean < dark)
    if not n:
        return np.zeros(gray.shape, bool)
    sides = (labels[0], labels[-1], labels[:, 0], labels[:, -1])
    touched = np.zeros(n + 1, int)
    for side in sides:
        touched[np.unique(side)] += 1
    touched[0] = 0
    keep = np.flatnonzero(touched >= min_edges)
    if not len(keep):
        return np.zeros(gray.shape, bool)
    return ndimage.binary_dilation(np.isin(labels, keep), iterations=size)
