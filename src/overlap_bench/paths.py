"""This project's own paths. Deliberately not named ``defns`` and not at
repo root: 07_posn also has a top-level ``defns.py``, and once its root is
on ``sys.path`` for ``reference_repo``, a same-named top-level module here
would collide with it. Living inside the ``overlap_bench`` package avoids
that without needing 08's own root on ``sys.path`` at all.
"""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
CAE_DIR = os.path.join(DATA_DIR, "cae")
REFERENCE_REPO = os.path.normpath(os.path.join(ROOT_DIR, "..", "07_posn"))
REFERENCE_COMMIT = "18a60649d927efb1504472cb33bd52b22db0f277"
EXPLORATION_LOG_REPO = os.path.normpath(
    os.path.join(ROOT_DIR, "..", "06_nb_mnist_phase_encoder")
)
