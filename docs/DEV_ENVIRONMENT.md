# The development environment: what this project runs on

mlws-ocr was built in a home office, by one person, on hardware you might
already own. Every network in it was trained here, from public data.
There were no cloud GPUs, no rented servers, no paid datasets and no paid
APIs. This page says exactly what the machines are, what each one does,
how long the work takes on them, and how to do the same with less.

The short version:

- **One desktop Mac** does everything: writing the code, running the
  engine, every evaluation, and the smaller training runs on its built-in
  GPU.
- **Two gaming PCs** from about 2021, each with one consumer NVIDIA
  graphics card, run the long training jobs and the big data harvests. They
  make training several times faster (ten times for the line reader).
  They are a convenience, not a requirement.
- **All the software is free**, and so is all the data.

## 1. The machines

| | the Mac | ai01 | ai02 |
|---|---|---|---|
| what it is | Mac mini (2024) | Alienware gaming desktop | ASUS ROG Strix gaming desktop |
| processor | Apple M4 Pro, 14 cores (10 performance, 4 efficiency) | Intel Core i7-11700KF, 8 cores / 16 threads | AMD Ryzen 9 5900X, 12 cores / 24 threads |
| memory | 48 GB (shared with its GPU) | 64 GB | 64 GB |
| graphics | the M4 Pro's built-in GPU (PyTorch's "MPS" backend) | NVIDIA GeForce RTX 3080 Ti, 12 GB | NVIDIA GeForce RTX 3090, 24 GB |
| storage | 1 TB SSD | 1 TB NVMe SSD | 1 TB NVMe SSD and a 2 TB hard disk |
| system | macOS | Ubuntu Linux 26.04 | Ubuntu Linux 26.04 |
| role | development, every evaluation, small training runs, figures | training, harvests over the big table datasets | training |

The two PCs sit on the home network and are used over `ssh` with no
screen attached. Neither GPU is a data-centre part. The RTX 3080 Ti and
RTX 3090 are graphics cards for games, two generations old as of 2026.
For the project's first three weeks (it began on 31 August 2026) all the
training ran on the Mac. ai01 joined on 21 September, ai02 on 6 October.
ai01 also needed a new CPU cooler along the way: its liquid cooler's pump
failed on 5 October, and until it was replaced the work moved to the
other two machines.

## 2. What each kind of work needs

**Running the engine.** The engine itself is numpy, scipy, scikit-image
and Pillow. It needs no GPU and no PyTorch. An evaluation process takes
about 2 GB of memory, and a page takes from a few seconds to about
fifteen. The Mac runs up to twelve evaluation processes side by side.
The full set of
scored table sets (seven sets, 280 tables) takes about 18 minutes, and the
1,750-table fitting set about an hour.

**Training.** Most trainers use PyTorch, an optional extra
(`pip install -e ".[train]"`), on whatever device it finds: an NVIDIA card
(CUDA), the Mac's GPU (MPS), or the processor. The networks are small, so
none of this needs a large card. The largest, the table sequence
reader, has 1.88 million parameters; a phone camera's face detector is
larger.

Some real training times, from [NETWORKS.md](NETWORKS.md):

| network | run | Apple GPU (MPS) | NVIDIA card |
|---|---|---|---|
| character language model | one epoch, pure numpy on the processor | under a minute | — |
| line reader (291k parameters) | eight epochs | about 11 hours | about 1 hour (RTX 3080 Ti) |
| equation reader (848k) | 30 epochs, 60,000 equations | 4.4 hours | 86 minutes (RTX 3090) |
| word-relation network (321k) | 20 epochs | — | about 45 minutes (RTX 3080 Ti) |
| table structure network (277k) | 12 epochs | — | 13 minutes an epoch (RTX 3080 Ti) |
| table sequence reader (1.88M) | 15 epochs, 385,953 tables, at half speed | — | 17 hours (RTX 3090) |

The Mac can run every one of these. A GPU card turns an overnight run
into an hour, which matters when you are trying ten variants in a week.

**Harvesting data.** Cutting training lines and tables out of the large
public datasets is processor work (numpy). It runs on the PCs, near the
data: PubTables-1M's training images alone fill about half of ai01's
1 TB disk. The repository's own `data/` directory on the Mac (models,
training files, evaluation sets, and the downloaded scans and receipts)
is about 165 GB.

## 3. Energy

The training scripts log the GPU's own power draw once a minute. The
table sequence reader's 17-hour run averaged 220 W at the card, about
3.8 kWh for the whole run; its 8-hour fine-tune averaged 211 W, about
1.6 kWh. The whole machine draws more than the card alone; we have not
metered it at the wall. Even so, the largest training run in the project
used about as much electricity as running a clothes dryer for an hour or
two.

The long runs are deliberately **gentle**: after each step the trainer rests
for half as long as the step took (`--duty 0.5`), runs at the lowest priority (`nice -n 19`), one job
per machine, and logs the processor's and the GPU's temperatures. The
GPU stayed near 60 °C (62 at most) and the processor under 71 °C. A run
takes half as long again, and the hardware is spared. Nobody's home machine should run flat out
for days.

## 4. The software

Everything is free and open source, and installs with Homebrew on the Mac
and `apt` or `pip` on Linux:

| tool | used for |
|---|---|
| Python 3.13 | everything |
| numpy, scipy, scikit-image, Pillow, pypdf | the engine |
| PyTorch 2.14 (optional) | training only; every network's weights are exported to a numpy `.npz` that the engine runs |
| pytest | the test suite (about 280 tests, two minutes on the Mac) |
| tectonic (LaTeX) and pdftoppm (Poppler) | typesetting the equation reader's training formulas |
| Google Chrome, headless | drawing the screenshot table set |
| Tesseract 5 | a comparison only; the engine does not use it |
| git and GitHub | the code, the issues, the released model files |

The NVIDIA cards need NVIDIA's Linux driver and a PyTorch build for CUDA.
Nothing else is special.

Development is a pair: the owner, and Claude Code (Anthropic's
coding assistant), which writes code and runs experiments under the
owner's direction. Commits made together carry its `Co-Authored-By` line.
Claude Code is not part of the engine. It trains nothing and is not a
model the engine uses ([NETWORKS.md](NETWORKS.md) has the rule: no
pre-trained models, no language or vision foundation models).

## 5. The data

Every data set is a free download, listed with its licence in
[DATA_SOURCES.md](DATA_SOURCES.md): the UNLV/ISRI scans, PubTables-1M,
FinTabNet.c, SROIE, FUNSD, CORD, Project Gutenberg's books for the
language model, and US government pages. Much of the training data is
not downloaded at all but drawn by the repository itself: text lines in
the computer's fonts, tables, equations, screenshots
([SYNTHETIC_DATA.md](SYNTHETIC_DATA.md); what it looks like:
[NEURAL_NETWORK_THEORY.md §0.2](NEURAL_NETWORK_THEORY.md#02-what-training-data-looks-like)).

## 6. Doing it with less

- **No GPU at all.** You can run the engine, the tests, every evaluation
  and the small trainers on a recent laptop. Training the larger networks
  on a processor alone works, but takes days, not hours. Use the released
  weights (`scripts/fetch_models.py`) and train only what you are
  changing.
- **An Apple-silicon Mac.** Everything here runs on one, including every
  trainer, through PyTorch's MPS backend: about ten times slower than a
  gaming card for the line reader, which is still an overnight run.
- **One used gaming card.** A 12 GB NVIDIA card trained the line readers,
  the table structure network and the word-relation network here. If you
  add one thing to a Mac or laptop, this is the one.
- **Less disk.** The big table datasets are only needed to retrain the
  table networks. The engine, its models and the evaluation sets fit in
  a few gigabytes.

## 7. Habits that keep it cheap

- **Small networks, chosen by measurement.** Each network is as small as
  the measurement allows. A network joins the engine only when the
  engine's own evaluation says it helps ([RESEARCH.md](RESEARCH.md)), so
  no compute is spent keeping a big model that does not earn its place.
- **Evaluate on the development machine.** The PCs train; the Mac judges.
  On the PCs' processors an evaluation runs about five times slower than
  on the M4 Pro, so moving it there would save nothing.
- **One job per training machine.** Each run's speed stays predictable,
  so its time can be planned, and the machine stays cool.
- **Checkpoints every epoch.** A run can stop for a reboot, a power cut or
  a better idea, and continue from where it was.
- **Announce the long runs.** Anything that will keep a machine busy for
  hours is agreed first, so the owner knows what the house is doing
  overnight.
