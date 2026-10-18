"""Build the HTML edition of the system paper (docs/papers/readable-self-trained-ocr.md).

The markdown carries figure markers as HTML comments (<!--FIG:key-->),
invisible on GitHub; here each marker becomes the figure (drawn by
scripts/make_system_figures.py, caption from CAPTIONS). The page's CSS is
the tables paper's; the builder itself is scripts/paper_html.py.
"""
from paper_html import PAPERS, build

CAPTIONS = {
    "pipeline": ("1", "The seventeen stage slots, and the learned models each profile plugs into them, coloured by the first profile that has them. pure fills the same slots with no network; classic adds the glyph MLP, the character GRU and the noisy channel; neural adds the line reader and its judge, the word-strip scorer, the word calibrator, the segmenter judge and the equation reader; neural-table adds the table networks."),
    "choice": ("2", "Two readers and a judge on one Library of Congress typescript page (public domain), read by the neural profile. Grey: both readings of the line agree. Blue: they differ and the judge took the line reader's. Orange: they differ and the judge kept the classic decoder's. Pale: the line reader declined. On the right, some of the disagreements with the reading taken in colour; the judge is not always right (it kept the classic decoder's 'specific conditions.\"', which drops the opening quote)."),
    "results": ("3", "The neural profile against both Tesseract engines on the same pages (Table 3). mlws-ocr at v0.18.20; Tesseract measured once, at v0.18.16, on the same pages. The two clean sets come first: pages no training of this project has touched, which since v0.18.17 have informed one decision (§6.1; Table 3b follows them release by release). The held-out rows were kept from decisions but not from the readers' training."),
}
ALT = {
    "pipeline": "A diagram of four rows of stage boxes from magnify to output, with coloured model names under the slots they plug into",
    "choice": "A typescript page with its lines boxed in grey, blue and orange, beside a panel of counts and example line crops with two readings each",
    "results": "Horizontal bar charts of character and word accuracy for nine sets, three engines each",
}

build(PAPERS / "readable-self-trained-ocr.md", PAPERS / "readable-self-trained-ocr.html", PAPERS / "img/system",
      CAPTIONS, ALT, "Readable and Self-Trained", head_from=PAPERS / "tables-from-rules.html")
