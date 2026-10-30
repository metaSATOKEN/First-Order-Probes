# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
import unittest
import numpy as np
import pr0008_shadow_methods as m


class Tests(unittest.TestCase):
    def test_signed_mean(self):
        a = np.zeros((3, 17)); a[:, 11] = [-1, 0, 1]; a[:, 12] = [-2, 0, 2]
        s, _ = m.batch_scores(a, [True, False, False], [True]*3)
        np.testing.assert_allclose(s, [1/3, 0, 0])

    def test_negative_cr_preserved(self):
        a = np.zeros((3, 17)); a[:, 11] = [-1, 0, 1]; a[:, 12] = [-4, -3, -2]
        s, d = m.batch_scores(a, [False]*3, [True]*3)
        self.assertLess(d['signed_T'][1], 0); self.assertEqual(s[2], 0)

    def test_zero_sd_separate(self):
        a = np.zeros((3, 17)); a[:, 11] = [-1, 0, 1]
        s, _ = m.batch_scores(a, [False]*3, [True]*3)
        self.assertTrue(np.isfinite(s[:2]).all()); self.assertTrue(np.isnan(s[2]))

    def test_nonfinite(self):
        a = np.zeros((3, 17)); a[0, 0] = np.nan
        s, _ = m.batch_scores(a, [False]*3, [True]*3)
        self.assertTrue(np.isnan(s).all())

    def test_heldout_not_in_threshold(self):
        clean = np.arange(25*6*3, dtype=float).reshape(25, 6, 3)
        c = m.fold_thresholds(clean)
        changed = clean.copy(); changed[:5] = 1e9
        np.testing.assert_array_equal(c[0], m.fold_thresholds(changed)[0])
        self.assertFalse(np.array_equal(c[1], m.fold_thresholds(changed)[1]))

    def test_strict_tie_and_unknown(self):
        s = np.ones((25, 6, 7, 3)); c = np.ones((5, 2, 3))
        self.assertTrue((m.gate_states(s, c) == 0).all())
        s[0, 0, 0, 0] = 2; s[1, 0, 0, 1] = np.nan
        x = m.gate_states(s, c)
        self.assertEqual(x[0, 0, 0, 0], 1); self.assertEqual(x[1, 0, 0, 1], 2)

    def test_development_invalid_propagates(self):
        a = np.ones((25, 6, 3)); a[0, 0, 0] = np.nan
        c = m.fold_thresholds(a)
        self.assertTrue(np.isfinite(c[0, 0, 0])); self.assertTrue(np.isnan(c[1:, 0, 0]).all())

    def test_split_disjoint(self):
        ids = np.arange(31999)[::-1]; u, o = ids[:15999], ids[15999:31998]
        self.assertEqual(len(np.intersect1d(u, o)), 0); self.assertEqual(len(o), 15999)
        self.assertGreater(len(np.intersect1d(u, ids[15998:31997])), 0)

    def test_decode_and_pairs(self):
        a = np.full((4, 2), int('888888', 16), np.uint32)
        b = a.copy(); a[:2] |= 1; b[1:3] |= 1
        np.testing.assert_array_equal(m.paired(a, b)[:, 0], [[1, 1, 1, 1]]*2)
        a[0, 0] |= 1 << 28
        with self.assertRaises(ValueError): m.decode(a)

    def test_policy_selection(self):
        old = np.ones((25, 6, 4, 2, 6), np.int64)
        u = np.full((25, 6, 7, 4, 2, 6), 3, np.int64)
        s = np.zeros((25, 6, 7, 3), np.uint8); s[..., 1] = 1; s[..., 2] = 2
        p = m.select_counts(old, u, s)
        self.assertTrue((p[..., 2] == 3).all()); self.assertTrue((p[..., 3:] == 1).all())


if __name__ == '__main__': unittest.main()
