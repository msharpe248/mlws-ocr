"""Locate usable TrueType fonts on the host system.

The factory renders training glyphs from real fonts; tests just need *a*
font, so discovery is deliberately dumb: look in the standard directories.
"""
from __future__ import annotations

from pathlib import Path

FONT_DIRS = [
    Path("/System/Library/Fonts/Supplemental"),  # macOS
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
    Path("/usr/share/fonts"),                    # Linux
    Path.home() / "Library/Fonts",
    Path.home() / ".fonts",
]


def find_fonts(pattern: str = "*.tt[fc]") -> list[Path]:
    """All TrueType files -- .ttc collections included (PIL loads face 0;
    Copperplate and friends ship only as .ttc on macOS)."""
    found: list[Path] = []
    for d in FONT_DIRS:
        if d.is_dir():
            found.extend(sorted(d.rglob(pattern)))
    return found


def default_font() -> Path:
    """A plain, ubiquitous font for tests and demos."""
    preferred = ["Arial.ttf", "Helvetica.ttf", "DejaVuSans.ttf",
                 "LiberationSans-Regular.ttf", "Verdana.ttf"]
    fonts = find_fonts()
    by_name = {f.name: f for f in fonts}
    for name in preferred:
        if name in by_name:
            return by_name[name]
    if fonts:
        return fonts[0]
    raise FileNotFoundError(f"no .ttf fonts found under {FONT_DIRS}")


NON_PRINT_HINTS = (
    "ornament", "wingding", "webding", "symbol", "dingbat", "emoji",
    "braille", "zapf", "smallcap", "bodoni 72", "hand", "chalk", "comic",
    "marker", "brush", "script", "sign", "noteworthy", "party",
    "trattatello", "papyrus", "narrow",
    # Decorative faces and non-Latin-primary families whose Latin glyphs
    # pass the shape check but poison the prototype set (found via the
    # UNLV real-scan study: median match distance 75 vs ~40 synthetic):
    "academy", "luminari", "kefa", "plantagenet", "keyboard",
    "applegothic", "applemyungjo", "ayuthaya", "khmer", "lao sangam",
    "krungthep", "sathu", "silom", "gothic neo", "hiragino", "pingfang",
    "herculanum",  # decorative roman caps; added confusable neighbors (measured)
    "sfcamera",    # symbol face, not text
)


def print_fonts(limit: int | None = None, exclude: tuple[str, ...] = ()) -> list[Path]:
    """Fonts suitable for printed-document glyph rendering.

    Name-filtered against decorative families, then shape-checked: 'o'
    must have a hole and 'l' must be a tall bar -- a face failing that is
    not rendering Latin text.
    """
    from mlws_ocr.glyph.features import FEATURE_NAMES, extract_features
    from .synth import render_glyph
    i_hole = FEATURE_NAMES.index("holes_r0")
    i_aspect = FEATURE_NAMES.index("aspect")

    out = []
    for f in find_fonts():
        name = f.name.lower()
        if any(e.lower() in name for e in exclude):
            continue
        # Display faces bypass the decorative exclusion AND the shape
        # gate: they are stocked deliberately for per-block routing, and
        # the gate exists to catch junk, not intentional stock (Impact's
        # tiny counters fail the o-has-a-hole test).
        if font_family(f) == "display":
            try:
                render_glyph("o", f, px_height=32)
            except Exception:
                continue
            out.append(f)
            if limit and len(out) >= limit:
                break
            continue
        if any(t in name for t in NON_PRINT_HINTS):
            continue
        try:
            o = extract_features(render_glyph("o", f, px_height=32))
            l = extract_features(render_glyph("l", f, px_height=32))
        except Exception:
            continue
        # aspect > 1.2 (not 1.5): serifed/typewriter 'l' has feet
        # (Courier New measures 1.35) but is still a bar.
        if o[i_hole] >= 1 and l[i_hole] == 0 and l[i_aspect] > 1.2:
            out.append(f)
        if limit and len(out) >= limit:
            break
    return out


FAMILY_HINTS = {
    "serif": ("times", "georgia", "caslon", "stix", "newyork", "new york",
              "palatino", "baskerville", "hoefler", "charter", "athelas",
              "cochin", "didot", "garamond", "book antiqua", "bookman"),
    "mono": ("courier", "andale", "monaco", "menlo", "consolas",
             "american typewriter", "prestige"),
    "sans": ("arial", "helvetica", "verdana", "tahoma", "trebuchet", "skia",
             "geneva", "din", "microsoft sans", "lucida grande", "avenir"),
    # Display faces exist ONLY for per-block routing: letterhead logo
    # lines vote "display" and match here; body blocks never see these
    # exemplars (stocking them unrouted measurably diluted accuracy).
    "display": ("impact", "herculanum", "academy", "luminari", "cooper",
                "copperplate"),
}


def font_family(path) -> str:
    """serif / sans / mono / other, from the face name."""
    name = Path(path).stem.lower()
    for family, hints in FAMILY_HINTS.items():
        if any(h in name for h in hints):
            return family
    return "other"
