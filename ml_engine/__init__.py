"""Machine-learning take/drop models for Nifty option trades."""
import os

# On this Windows setup, scikit-learn's OpenMP thread pool takes ~40 s to start on the first
# gradient-boosting fit; a single thread starts instantly and these small models stay fast.
# Must be set before scikit-learn is first imported, which happens through this package.
os.environ.setdefault("OMP_NUM_THREADS", "1")
