"""Histogram and Jensen-Shannon divergence (distances.py)."""
import numpy as np
import pytest

import distances as dist


def test_histogram_sums_to_one_and_bins_edges():
    h = dist.histogram(np.array([0.0, 0.05, 0.95, 1.0]))
    assert h.sum() == pytest.approx(1.0)
    # 0.0 and 0.05 land in the first bin; 1.0 is included in the last bin
    assert h[0] == pytest.approx(0.5) and h[-1] == pytest.approx(0.5)


def test_histogram_empty_is_all_zero():
    assert dist.histogram(np.array([])).sum() == 0


def test_jsd_bounds():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert dist.jsd(p, p) == pytest.approx(0.0)
    assert dist.jsd(p, q) == pytest.approx(1.0)     # disjoint -> maximum, base 2
    assert dist.jsd(p, q) == pytest.approx(dist.jsd(q, p))   # symmetric


def test_jsd_half_overlap():
    # p = [1, 0], q = [.5, .5]: m = [.75, .25]
    # JSD = .5*KL(p||m) + .5*KL(q||m) = .5*log2(4/3) + .5*(.5*log2(2/3) + .5*log2(2))
    expected = 0.5 * np.log2(4 / 3) + 0.5 * (0.5 * np.log2(2 / 3) + 0.5)
    assert dist.jsd(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == pytest.approx(expected)


def test_compute_jsd():
    human = {1: np.array([0.05, 0.05]), 2: np.array([0.95, 0.95]), 3: np.array([0.05, 0.95])}
    synth = {1: np.array([0.05, 0.05]), 2: np.array([0.05, 0.05]), 9: np.array([0.5])}
    df, _ = dist.compute(human, synth)
    df = df.set_index("pid")
    assert 9 not in df.index                      # no human twin -> skipped
    assert df.loc[1, "JSD"] == pytest.approx(0.0)
    assert df.loc[2, "JSD"] == pytest.approx(1.0)
    assert df.loc[1, "n_question"] == 2
