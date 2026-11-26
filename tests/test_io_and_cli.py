"""Тесты ввода-вывода и интерфейса командной строки."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

import generate_data
import main
from io_utils import (CSV_COLUMNS, ImageSeriesError, append_metrics, list_images,
                      read_gray, read_series, save_image)


class TestIO(unittest.TestCase):
    """Чтение и запись изображений."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_folder(self):
        with self.assertRaises(ImageSeriesError):
            list_images(self.root / "нет_такой")

    def test_empty_folder(self):
        with self.assertRaises(ImageSeriesError):
            list_images(self.root)

    def test_unsupported_files_ignored(self):
        (self.root / "notes.txt").write_text("привет")
        with self.assertRaises(ImageSeriesError):
            list_images(self.root)

    def test_broken_image(self):
        bad = self.root / "bad.png"
        bad.write_bytes(b"not an image")
        with self.assertRaises(ImageSeriesError):
            read_gray(bad)

    def test_missing_file(self):
        with self.assertRaises(ImageSeriesError):
            read_gray(self.root / "нет.png")

    def test_roundtrip_with_russian_path(self):
        image = np.random.default_rng(0).integers(0, 255, (20, 30), dtype=np.uint8)
        path = self.root / "папка" / "кадр.png"
        save_image(path, image)
        self.assertTrue(np.array_equal(read_gray(path), image))

    def test_different_sizes(self):
        save_image(self.root / "a.png", np.zeros((10, 10), np.uint8))
        save_image(self.root / "b.png", np.zeros((12, 10), np.uint8))
        with self.assertRaises(ImageSeriesError):
            read_series(self.root)

    def test_max_frames(self):
        for i in range(4):
            save_image(self.root / f"f{i}.png", np.zeros((5, 5), np.uint8))
        names, frames = read_series(self.root, max_frames=2)
        self.assertEqual(names, ["f0", "f1"])

    def test_csv_header_written_once(self):
        path = self.root / "m.csv"
        append_metrics(path, {"series": "a"})
        append_metrics(path, {"series": "b"})
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0].split(","), CSV_COLUMNS)


class TestCLI(unittest.TestCase):
    """Запуск программы целиком."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            code = generate_data.main(["--output", str(cls.root / "кадры"), "--frames", "6",
                                       "--reference", str(cls.root / "ref.png")])
        assert code == 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_main(self, *args):
        """Запускает main и возвращает (код, текст ошибок)."""
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            try:
                code = main.main(list(args))
            except SystemExit as exit_info:  # argparse завершает программу сам
                code = exit_info.code
        return code, err.getvalue()

    def test_successful_run(self):
        out = self.root / "результаты"
        code, _ = self.run_main("--input", str(self.root / "кадры"), "--output", str(out),
                                "--reference", str(self.root / "ref.png"))
        self.assertEqual(code, 0)
        self.assertTrue((out / "result_rigid_mean.png").exists())
        self.assertTrue((out / "metrics.csv").exists())
        self.assertEqual(len(list((out / "masks").glob("*_mask.png"))), 6)
        self.assertTrue((out / "contours" / "frame_00_contour.jpg").exists())

    def test_missing_input_folder(self):
        code, err = self.run_main("--input", str(self.root / "нет"))
        self.assertEqual(code, 1)
        self.assertIn("Каталог не найден", err)

    def test_bad_sigma(self):
        code, _ = self.run_main("--input", str(self.root / "кадры"), "--sigma", "-1")
        self.assertEqual(code, 2)

    def test_bad_threshold(self):
        code, _ = self.run_main("--input", str(self.root / "кадры"), "--threshold", "999")
        self.assertEqual(code, 2)

    def test_bad_method(self):
        code, _ = self.run_main("--input", str(self.root / "кадры"), "--method", "magic")
        self.assertEqual(code, 2)

    def test_single_frame_is_error(self):
        code, err = self.run_main("--input", str(self.root / "кадры"), "--max-frames", "1",
                                  "--output", str(self.root / "o1"))
        self.assertEqual(code, 1)
        self.assertIn("минимум два кадра", err)

    def test_wrong_reference_size(self):
        save_image(self.root / "small.png", np.zeros((10, 10), np.uint8))
        code, err = self.run_main("--input", str(self.root / "кадры"), "--reference",
                                  str(self.root / "small.png"), "--output", str(self.root / "o2"))
        self.assertEqual(code, 1)
        self.assertIn("эталона", err)


if __name__ == "__main__":
    unittest.main()
