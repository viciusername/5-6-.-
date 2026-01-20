"""Package for Lab 5+6.

We force a non-interactive Matplotlib backend so plots can be saved everywhere
(Windows/Linux/headless).
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
