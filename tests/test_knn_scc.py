"""knn_scc's 2026-09-25 options: block order by XY-cut, gutter fences, the
per-block gutter test, the level tree, the learned link features, and the
vectorised neighbour search (which must reproduce the loop exactly)."""
import numpy as np

from mlws_ocr.layout import knn_scc as K


def test_xycut_order_reads_columns_before_rows():
    # a spanning headline, then two columns of two paragraphs each whose
    # paragraph breaks line up across the gutter
    head = [0, 0, 1000, 80]
    l1, l2 = [0, 100, 480, 500], [0, 520, 480, 900]
    r1, r2 = [520, 100, 1000, 500], [520, 520, 1000, 900]
    blocks = [r2, l1, head, r1, l2]
    assert K.order_xycut(blocks, "v") == [head, l1, l2, r1, r2]
    # top-left order interleaves the columns
    assert sorted(blocks, key=lambda b: (b[1], b[0])) == [head, l1, r1, l2, r2]
    # horizontal-first reads row by row
    assert K.order_xycut(blocks, "h") == [head, l1, r1, l2, r2]


def test_crosses_gutter_only_for_links_that_jump_it():
    centers = np.array([[100.0, 500.0], [300.0, 500.0], [150.0, 520.0], [300.0, 2000.0]])
    edges = np.array([[0, 1], [0, 2], [1, 0], [0, 3]])
    gutters = np.array([[200.0, 100.0, 240.0, 1000.0]])
    got = K.crosses_gutter(edges, centers, gutters)
    # 0-1 jumps it both ways; 0-2 stays left; 0-3 spans x and overlaps it vertically
    assert got.tolist() == [True, False, True, True]
    assert not K.crosses_gutter(edges, centers, np.zeros((0, 4))).any()


def _column(x0, x1, y0, n, pitch=40, h=20, w=15):
    return [[x, y0 + k * pitch, x + w, y0 + k * pitch + h]
            for k in range(n) for x in range(x0, x1 - w, 18)]


def test_local_gutter_finds_the_empty_run_between_two_columns():
    two = np.array(_column(0, 400, 0, 20) + _column(460, 860, 0, 20))
    g = K.local_gutter(two, min_w=16)
    assert g is not None and 380 <= g[0] and g[2] <= 470
    one = np.array(_column(0, 860, 0, 20))
    assert K.local_gutter(one, min_w=16) is None


def test_level_groups_split_only_where_a_gutter_lies_inside():
    boxes = np.array(_column(0, 400, 0, 10) + _column(460, 860, 0, 10), float)
    n = len(boxes)
    half = n // 2
    # coarse: everything one component; fine: the two columns apart
    coarse = np.zeros(n, int)
    fine = np.array([0] * half + [1] * (n - half))
    labels = iter([coarse, fine])

    def scc(mask):
        lab = next(labels)
        return int(lab.max()) + 1, lab

    gutters = np.array([[410.0, 0.0, 450.0, 400.0]])
    groups = K.level_groups([None, None], scc, boxes, gutters)
    assert sorted(len(g) for g in groups) == [half, n - half]
    labels = iter([coarse, fine])
    groups = K.level_groups([None, None], scc, boxes, np.zeros((0, 4)))
    assert [len(g) for g in groups] == [n]      # no gutter: the coarse level stands


def _loop_edges(centers, k_per_dir, k_total, exempt):
    tree = K.cKDTree(centers)
    dists, idxs = tree.query(centers, k=min(len(centers), 40))
    edges, lengths = [], []
    for i in range(len(centers)):
        cand = {d: [] for d in range(8)}
        for d, j in zip(dists[i][1:], idxs[i][1:]):
            if j == i or not np.isfinite(d):
                continue
            dx, dy = centers[j] - centers[i]
            cand[int(((np.arctan2(dy, dx) + np.pi) / (np.pi / 4))) % 8].append((float(d), int(j)))
        picks = [(d, j) for items in cand.values() for d, j in sorted(items)[:k_per_dir]]
        if k_total is not None and not exempt[i]:
            picks = sorted(picks)[:k_total]
        for d, j in picks:
            edges.append((i, j)); lengths.append(d)
    return np.array(edges), np.array(lengths)


def test_vectorised_neighbour_search_matches_the_loop():
    rng = np.random.default_rng(3)
    centers = np.round(rng.uniform(0, 500, (300, 2)))      # rounding makes distance ties
    exempt = rng.uniform(size=300) < 0.1
    for k_total in (None, 5, 3):
        e0, l0 = _loop_edges(centers, 3, k_total, exempt)
        e1, l1 = K.directional_edges(centers, 3, k_total=k_total, pool_exempt=exempt)
        assert np.array_equal(e0, e1) and np.array_equal(l0, l1)


def test_link_features_one_row_per_link():
    boxes = np.array(_column(0, 400, 0, 5), float)
    centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])
    sizes = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    edges, lengths = K.directional_edges(centers, 3)
    X = K.link_features(edges, lengths, centers, boxes, sizes, 20.0, np.zeros(len(edges), bool))
    assert X.shape == (len(edges), len(K.LINK_FEATURES)) and np.isfinite(X).all()


def test_join_display_rows_joins_a_headline_row_but_never_over_a_column():
    ref = 20.0
    # a headline of three 80-px letters, then a body column of 20-px glyphs below
    letters = [[0, 0, 60, 80], [70, 0, 130, 80], [140, 0, 200, 80]]
    body = [[x, y, x + 15, y + 20] for y in range(120, 400, 40) for x in range(0, 200, 18)]
    boxes = np.array(letters + body, float)
    sizes = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    column = [0, 120, 200, 400]
    out = K.join_display_rows(letters + [column], boxes, sizes, ref)
    assert [0, 0, 200, 80] in out and column in out and len(out) == 2
    # a block reaching up between the letters would be overlapped: no join
    intruder = [62, 10, 68, 70]
    out = K.join_display_rows(letters + [intruder], boxes, sizes, ref)
    assert len(out) == 4


def test_judged_stage_returns_the_chosen_candidates_blocks(tmp_path):
    """Every candidate must write its own layout: the judge's pick is what
    comes out, not the last candidate run."""
    from mlws_ocr.core.artifacts import Page
    from mlws_ocr.layout import segjudge
    rng = np.random.default_rng(0)
    binary = np.zeros((600, 800), bool)
    for y in range(50, 550, 30):
        for x in list(range(40, 360, 14)) + list(range(440, 760, 14)):
            binary[y:y + 12, x:x + 8] = rng.uniform() < 0.9
    page = Page(gray=(~binary).astype(np.float32), binary=binary, dpi=300.0,
                meta={"doc_type": "newspaper", "layout": {}})
    cands = ["xyH", "tree"]
    ctx_n = 7
    for pick in cands:
        w = np.zeros(len(cands) * ctx_n + 4)
        w[cands.index(pick) * ctx_n] = 10.0          # bias towards `pick`
        path = tmp_path / f"judge_{pick}.npz"
        np.savez(path, w=w, cands=np.array(cands), hint=True)
        out, dbg = segjudge.JudgedBlocks(model_path=str(path)).run(page)
        alone = segjudge.run_candidate(page, pick)[0].meta["layout"]["blocks"]
        assert dbg.scalars["chosen"] == pick and out.meta["layout"]["blocks"] == alone


def test_judged_stage_is_xycut_on_pages_it_does_not_judge():
    from mlws_ocr.core.artifacts import Page
    from mlws_ocr.core.registry import get
    from mlws_ocr.layout import segjudge  # noqa: F401  (registers "judged")
    rng = np.random.default_rng(1)
    binary = np.zeros((500, 700), bool)
    for y in range(40, 460, 28):
        for x in range(40, 660, 13):
            binary[y:y + 11, x:x + 7] = rng.uniform() < 0.85
    for doc_type in ("letter", "legal", None):
        meta = {"layout": {}} | ({"doc_type": doc_type} if doc_type else {})
        page = Page(gray=(~binary).astype(np.float32), binary=binary, dpi=300.0, meta=meta)
        judged = get("blocks", "judged")(model_path="/nonexistent").run(page)[0].meta["layout"]["blocks"]
        page2 = Page(gray=(~binary).astype(np.float32), binary=binary, dpi=300.0, meta=dict(meta, layout={}))
        assert judged == get("blocks", "xycut")().run(page2)[0].meta["layout"]["blocks"]
