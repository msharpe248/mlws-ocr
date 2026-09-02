"""Outline-segment features matched many-to-one against prototype segments.

This is a readable re-derivation of the mechanism Ray Smith credits for
Tesseract's tolerance of damaged print ("An Overview of the Tesseract OCR
Engine", ICDAR 2007, §5.1-5.2; code: src/classify/intfx.cpp, intmatcher.cpp).
Our other channels -- the 95-vector prototypes and the MLP over them --
describe a glyph as a WHOLE, so a broken stem or an eroded bowl moves every
feature at once.  Here a glyph is a cloud of short oriented pieces of its
outline, and a class is a small set of configurations (one per training
glyph), each a list of longer straight prototype segments.  Evidence is
accumulated pairwise, so a break costs one unmatched prototype and a few
unmatched features while everything else still matches.

Pipeline of one match:

  1. Character normalization.  Translate the glyph so its ink centroid is
     at the origin and scale each axis by its second moment (Tesseract's
     "moment normalization": removes font aspect ratio and much of stroke
     width; case then rests on the decoder's height priors, as before).
  2. Features.  Walk each outline (outer and hole, oriented) in normalized
     space and cut it into pieces of a fixed length; each piece is a
     feature (x, y, theta) -- position of its midpoint and direction.
  3. Prototypes.  A training glyph's outlines, polygon-approximated in the
     same space, give segments (x0, y0, x1, y1); a segment's LENGTH says
     how many features it may explain.
  4. Evidence.  e(f, p) = exp(-(d/sigma_d)^2 - (dtheta/sigma_theta)^2), d the
     distance from the feature point to the segment, dtheta the direction
     difference.  Per configuration: each feature keeps its best prototype
     (sum -> feature evidence); each prototype keeps its best L matches, L
     its length in feature units (sum -> prototype evidence); rating =
     (both sums) / (n_features + sum of L) -- Tesseract's NormalizeSums.
     Cost = 1 - best configuration rating.

Only the candidate classes the other channels propose are rated (a
second opinion), so a glyph costs roughly 60 features x a few hundred
segments.
"""
from __future__ import annotations

import numpy as np
from skimage import measure

FEATURE_LEN = 12.8          # normalized units (Tesseract: 64/5 in a 256 space)
SCALE = 51.2                # second moment -> this many units (Tesseract)
POLY_TOL = 3.0              # polygon approximation tolerance, normalized units


def _normalizer(mask: np.ndarray):
    """(cy, cx, sy, sx): centroid and per-axis scale of the ink."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return 0.0, 0.0, 1.0, 1.0
    cy, cx = ys.mean(), xs.mean()
    ry = max(float(np.sqrt(((ys - cy) ** 2).mean())), 0.5)
    rx = max(float(np.sqrt(((xs - cx) ** 2).mean())), 0.5)
    return cy, cx, SCALE / ry, SCALE / rx


def _contours(mask: np.ndarray) -> list[np.ndarray]:
    """Closed outlines (row, col) of a binary glyph, consistently oriented
    (skimage keeps ink on one fixed side of every contour, so outer
    outlines and holes run in opposite directions -- a real cue)."""
    padded = np.pad(mask.astype(np.float32), 1)
    return [c - 1.0 for c in measure.find_contours(padded, 0.5)
            if len(c) >= 4]


def _normalized_outlines(mask: np.ndarray) -> list[np.ndarray]:
    cy, cx, sy, sx = _normalizer(mask)
    out = []
    for c in _contours(mask):
        pts = np.column_stack([(c[:, 1] - cx) * sx, (c[:, 0] - cy) * sy])
        out.append(pts)                                   # (x, y) columns
    return out


def outline_features(mask: np.ndarray, feature_len: float = FEATURE_LEN,
                     cut_edges: tuple[str, ...] = ()) -> np.ndarray:
    """(N, 3) features: x, y, theta (radians) of fixed-length outline pieces.

    cut_edges: "left" and/or "right" when the mask is a PIECE of a chopped
    blob.  The cut itself is a straight vertical run of outline that no
    prototype has, and it would be charged as unmatched evidence; features
    within two pixels of that edge are dropped so the piece is rated on
    the sides it really has (the many-to-one matching then costs the
    missing stretch only as uncovered prototype length).
    """
    feats = []
    cy, cx, sy, sx = _normalizer(mask)
    w = mask.shape[1]
    x_lo = (2.0 - cx) * sx if "left" in cut_edges else -np.inf
    x_hi = (w - 3.0 - cx) * sx if "right" in cut_edges else np.inf
    for pts in _normalized_outlines(mask):
        seg = np.diff(pts, axis=0)
        lens = np.hypot(seg[:, 0], seg[:, 1])
        total = lens.sum()
        n = int(round(total / feature_len))
        if n == 0:
            continue
        cum = np.concatenate([[0.0], np.cumsum(lens)])
        step = total / n
        for k in range(n):
            s0, s1 = k * step, (k + 1) * step
            p0 = _point_at(pts, cum, s0)
            p1 = _point_at(pts, cum, s1)
            d = p1 - p0
            if not d.any():
                continue
            mx = (p0[0] + p1[0]) / 2
            if mx < x_lo or mx > x_hi:
                continue                    # on a cut edge
            feats.append([mx, (p0[1] + p1[1]) / 2,
                          float(np.arctan2(d[1], d[0]))])
    return np.array(feats, np.float32).reshape(-1, 3)


def _point_at(pts, cum, s):
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = min(max(i, 0), len(pts) - 2)
    span = cum[i + 1] - cum[i]
    t = 0.0 if span <= 0 else (s - cum[i]) / span
    return pts[i] + t * (pts[i + 1] - pts[i])


def outline_prototypes(mask: np.ndarray, tol: float = POLY_TOL) -> np.ndarray:
    """(M, 4) prototype segments x0, y0, x1, y1 from a polygon approximation
    of each normalized outline (one training glyph = one configuration)."""
    segs = []
    for pts in _normalized_outlines(mask):
        closed = np.vstack([pts, pts[:1]])
        poly = measure.approximate_polygon(closed, tolerance=tol)
        for a, b in zip(poly[:-1], poly[1:]):
            if np.hypot(*(b - a)) >= 1.0:
                segs.append([a[0], a[1], b[0], b[1]])
    return np.array(segs, np.float32).reshape(-1, 4)


def evidence(feats: np.ndarray, segs: np.ndarray, sigma_d: float,
             sigma_t: float) -> np.ndarray:
    """(N, M) pairwise evidence between features and prototype segments."""
    if len(feats) == 0 or len(segs) == 0:
        return np.zeros((len(feats), len(segs)), np.float32)
    P = feats[:, None, :2]                          # N,1,2
    A = segs[None, :, :2]                           # 1,M,2
    B = segs[None, :, 2:]
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(-1), 1e-6)         # 1,M
    t = np.clip(((P - A) * AB).sum(-1) / L2, 0.0, 1.0)   # N,M projection
    closest = A + t[..., None] * AB
    d = np.hypot(*(P - closest).transpose(2, 0, 1))  # N,M
    theta_p = np.arctan2(AB[..., 1], AB[..., 0])     # 1,M
    dt = np.angle(np.exp(1j * (feats[:, None, 2] - theta_p)))   # wrapped
    return np.exp(-(d / sigma_d) ** 2 - (dt / sigma_t) ** 2).astype(np.float32)


def _proto_lengths(segs: np.ndarray, feature_len: float) -> np.ndarray:
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return np.maximum(np.round(lengths / feature_len).astype(int), 1)


def config_rating(feats: np.ndarray, segs: np.ndarray, sigma_d: float,
                  sigma_t: float, feature_len: float = FEATURE_LEN) -> float:
    """Tesseract's normalized evidence for one configuration, in [0, 1]."""
    if len(feats) == 0 or len(segs) == 0:
        return 0.0
    E = evidence(feats, segs, sigma_d, sigma_t)
    L = _proto_lengths(segs, feature_len)
    return float(_rating_from_evidence(E, L, [np.arange(len(segs))])[0])


def _rating_from_evidence(E: np.ndarray, L: np.ndarray,
                          groups: list[np.ndarray]) -> np.ndarray:
    """Ratings of several configurations sharing one evidence matrix.

    E is (n_features, n_segments) over the segments of ALL configurations
    concatenated; `groups` lists each configuration's column indices.
    Per configuration: feature evidence = sum over features of the best
    segment; prototype evidence = per segment, the sum of its best L
    matches (a segment of length L may explain at most L features);
    rating = (both) / (n_features + sum L).
    """
    n = E.shape[0]
    # per-segment top-L sums, vectorized: sort each column descending,
    # cumulative-sum, read the (L-1)th row
    srt = -np.sort(-E, axis=0)
    csum = np.cumsum(srt, axis=0)
    k = np.minimum(L, n) - 1
    proto_ev = csum[k, np.arange(E.shape[1])]
    # configurations are contiguous column ranges: reduce them all at once
    starts = np.array([g[0] for g in groups])
    feat_ev = np.maximum.reduceat(E, starts, axis=1).sum(axis=0)
    p_ev = np.add.reduceat(proto_ev, starts)
    l_sum = np.add.reduceat(L, starts)
    return ((feat_ev + p_ev) / (n + l_sum)).astype(np.float32)


class OutlineMatcher:
    """Per-class configurations of prototype segments; ratings on demand."""

    def __init__(self, sigma_d: float = 35.0, sigma_t: float = 0.7):
        # Widths swept on a broken-'h' study (cross-font clean and cut
        # glyphs against h/n/b/k): tight widths (10, 0.45) rated every
        # class near 0.2 and let 'b' win the cut 'h'; at (35-40, 0.7-0.8)
        # the cut 'h' rates 0.75 as 'h' against 0.71 as 'b', and a clean
        # cross-font 'h' 0.81 against 0.73.
        self.sigma_d, self.sigma_t = sigma_d, sigma_t
        self.configs: dict[str, list[np.ndarray]] = {}

    def add(self, cls: str, mask: np.ndarray) -> None:
        segs = outline_prototypes(mask)
        if len(segs):
            self.configs.setdefault(cls, []).append(segs)

    def rating(self, feats: np.ndarray, cls: str) -> float:
        """Best configuration rating for the class (0 if the class is unknown).
        All configurations of the class share one evidence matrix."""
        cfgs = self.configs.get(cls, [])
        if not cfgs or len(feats) == 0:
            return 0.0
        segs = np.concatenate(cfgs)
        groups, start = [], 0
        for c in cfgs:
            groups.append(np.arange(start, start + len(c)))
            start += len(c)
        E = evidence(feats, segs, self.sigma_d, self.sigma_t)
        L = _proto_lengths(segs, FEATURE_LEN)
        return float(_rating_from_evidence(E, L, groups).max())

    def costs(self, mask: np.ndarray, classes: list[str],
              cut_edges: tuple[str, ...] = ()) -> dict[str, float]:
        """1 - rating for each requested class."""
        feats = outline_features(mask, cut_edges=cut_edges)
        return {c: 1.0 - self.rating(feats, c) for c in classes}

    def costs_all(self, mask: np.ndarray,
                  cut_edges: tuple[str, ...] = ()) -> dict[str, float]:
        """1 - rating for EVERY class (used for chopped pieces, whose
        whole-glyph candidate lists are unreliable)."""
        return self.costs(mask, list(self.configs), cut_edges)

    # -- storage: ragged configurations flattened with index arrays
    def save(self, path) -> None:
        segs, cls_idx, cfg_idx, classes = [], [], [], sorted(self.configs)
        for ci, c in enumerate(classes):
            for k, s in enumerate(self.configs[c]):
                segs.append(s)
                cls_idx.extend([ci] * len(s))
                cfg_idx.extend([k] * len(s))
        np.savez_compressed(path, segs=np.concatenate(segs), cls=np.array(cls_idx),
                            cfg=np.array(cfg_idx), classes=np.array(classes),
                            sigma=np.array([self.sigma_d, self.sigma_t]))

    @classmethod
    def load(cls, path) -> "OutlineMatcher":
        d = np.load(path, allow_pickle=False)
        m = cls(float(d["sigma"][0]), float(d["sigma"][1]))
        classes = [str(c) for c in d["classes"]]
        segs, ci, ki = d["segs"], d["cls"], d["cfg"]
        for c_i in np.unique(ci):
            sel = ci == c_i
            for k in np.unique(ki[sel]):
                m.configs.setdefault(classes[c_i], []).append(segs[sel & (ki == k)])
        return m
