"""Тесты для processing.py и metrics.py."""

import math
import unittest

import cv2
import numpy as np

import metrics
from generate_data import make_clean, make_series
from processing import (ContourNotFoundError, alignment_matrix, angle_difference,
                        combine_frames, contour_features, filled_mask, find_contour,
                        mask_iou, process_series, segment, warp)


def shifted(image, dx, dy, angle=0.0):
    """Сдвигает и поворачивает изображение вокруг центра (для тестов)."""
    h, w = image.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    matrix[:, 2] += (dx, dy)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


class TestSegmentation(unittest.TestCase):
    """Поиск контура и его характеристик."""

    def setUp(self):
        self.clean = make_clean()

    def test_finds_object(self):
        mask = segment(self.clean)
        contour = find_contour(mask)
        self.assertGreater(cv2.contourArea(contour), 10000)

    def test_dark_object_on_light_background(self):
        inverted = 255 - self.clean
        area_a = cv2.contourArea(find_contour(segment(self.clean)))
        area_b = cv2.contourArea(find_contour(segment(inverted)))
        self.assertAlmostEqual(area_a, area_b, delta=area_a * 0.02)

    def test_empty_frame_raises(self):
        with self.assertRaises(ContourNotFoundError):
            find_contour(segment(np.full((100, 100), 50, np.uint8)))

    def test_centroid_moves_with_shift(self):
        a = contour_features(find_contour(segment(self.clean)))
        b = contour_features(find_contour(segment(shifted(self.clean, 10, -7))))
        self.assertAlmostEqual(b.cx - a.cx, 10, delta=0.5)
        self.assertAlmostEqual(b.cy - a.cy, -7, delta=0.5)

    def test_filled_mask_has_no_holes(self):
        contour = find_contour(segment(self.clean))
        mask = filled_mask(contour, self.clean.shape)
        self.assertEqual(mask[110, 165], 255)  # тёмное пятно внутри объекта

    def test_angle_difference_wraps(self):
        self.assertAlmostEqual(angle_difference(math.pi / 2 - 0.1, -math.pi / 2 + 0.1), -0.2)


class TestAlignment(unittest.TestCase):
    """Выравнивание кадров по контуру."""

    def setUp(self):
        self.clean = make_clean()
        self.ref = contour_features(find_contour(segment(self.clean)))

    def _iou_after(self, image, method):
        feat = contour_features(find_contour(segment(image)))
        matrix = alignment_matrix(feat, self.ref, method)
        aligned = warp(image, matrix)
        return mask_iou(filled_mask(find_contour(segment(self.clean)), image.shape),
                        filled_mask(find_contour(segment(aligned)), image.shape))

    def test_translate_fixes_shift(self):
        moved = shifted(self.clean, 12, 9)
        self.assertGreater(self._iou_after(moved, "translate"), 0.98)

    def test_rigid_fixes_shift_and_rotation(self):
        moved = shifted(self.clean, -8, 6, angle=4.0)
        self.assertGreater(self._iou_after(moved, "rigid"), 0.98)
        # сдвиг без поворота этого не исправит
        self.assertLess(self._iou_after(moved, "translate"), self._iou_after(moved, "rigid"))

    def test_none_keeps_image(self):
        moved = shifted(self.clean, 5, 5)
        feat = contour_features(find_contour(segment(moved)))
        matrix = alignment_matrix(feat, self.ref, "none")
        self.assertTrue(np.array_equal(warp(moved, matrix), moved))

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            alignment_matrix(self.ref, self.ref, "magic")


class TestCombine(unittest.TestCase):
    """Усреднение кадров."""

    def test_mean(self):
        frames = [np.full((4, 4), 10, np.uint8), np.full((4, 4), 20, np.uint8)]
        self.assertTrue((combine_frames(frames, "mean") == 15).all())

    def test_no_uint8_overflow(self):
        frames = [np.full((2, 2), 200, np.uint8)] * 3
        self.assertTrue((combine_frames(frames, "mean") == 200).all())

    def test_median_ignores_outlier(self):
        frames = [np.full((2, 2), 100, np.uint8), np.full((2, 2), 100, np.uint8),
                  np.full((2, 2), 255, np.uint8)]
        self.assertTrue((combine_frames(frames, "median") == 100).all())

    def test_empty_and_unknown(self):
        with self.assertRaises(ValueError):
            combine_frames([])
        with self.assertRaises(ValueError):
            combine_frames([np.zeros((2, 2), np.uint8)], "max")


class TestPipeline(unittest.TestCase):
    """Весь конвейер на синтетической серии."""

    @classmethod
    def setUpClass(cls):
        cls.clean, cls.frames, _ = make_series(n=8, seed=1)

    def test_averaging_reduces_noise(self):
        result = process_series(self.frames, "rigid", "mean")
        mask = result.masks[0]
        self.assertLess(metrics.noise_std(result.result, mask),
                        metrics.noise_std(self.frames[0], mask) / 2)

    def test_alignment_improves_quality(self):
        psnr = {m: metrics.psnr(process_series(self.frames, m, "mean").result, self.clean)
                for m in ("none", "translate", "rigid")}
        self.assertLess(psnr["none"], psnr["translate"])
        self.assertLess(psnr["translate"], psnr["rigid"])

    def test_reproducible_generator(self):
        _, again, _ = make_series(n=8, seed=1)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(self.frames, again)))

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            process_series(self.frames, "rigid", "mean", sigma=0)
        with self.assertRaises(ValueError):
            process_series(self.frames, "rigid", "mean", threshold=300)
        with self.assertRaises(ValueError):
            process_series([], "rigid", "mean")

    def test_blank_frame_reports_index(self):
        frames = list(self.frames)
        frames[3] = np.full_like(frames[0], 50)
        with self.assertRaises(ContourNotFoundError) as ctx:
            process_series(frames)
        self.assertIn("№3", str(ctx.exception))


class TestMetrics(unittest.TestCase):
    """Метрики качества."""

    def setUp(self):
        self.a = np.zeros((10, 10), np.uint8)
        self.b = np.full((10, 10), 10, np.uint8)

    def test_mse(self):
        self.assertEqual(metrics.mse(self.a, self.a), 0)
        self.assertEqual(metrics.mse(self.a, self.b), 100)

    def test_mse_no_overflow(self):
        # 0 - 200 в uint8 дало бы 56, а не -200
        self.assertEqual(metrics.mse(self.a, np.full((10, 10), 200, np.uint8)), 40000)

    def test_psnr(self):
        self.assertEqual(metrics.psnr(self.a, self.a), float("inf"))
        self.assertAlmostEqual(metrics.psnr(self.a, self.b), 10 * math.log10(255 ** 2 / 100))

    def test_ssim_identical(self):
        image = make_clean()
        self.assertAlmostEqual(metrics.ssim(image, image), 1.0)

    def test_size_mismatch(self):
        with self.assertRaises(ValueError):
            metrics.mse(self.a, np.zeros((5, 5), np.uint8))

    def test_noise_std_on_background(self):
        clean = make_clean()
        mask = segment(clean)
        self.assertEqual(metrics.noise_std(clean, mask), 0.0)
        noisy = clean.astype(np.float32) + np.random.default_rng(0).normal(0, 10, clean.shape)
        self.assertAlmostEqual(metrics.noise_std(np.clip(noisy, 0, 255).astype(np.uint8), mask),
                               10, delta=1)


if __name__ == "__main__":
    unittest.main()
