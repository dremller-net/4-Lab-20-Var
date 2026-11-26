"""Точка входа: пакетная обработка серии кадров (выравнивание по контуру и усреднение).

Пример запуска::

    python main.py --input images --output results --method rigid --combine mean
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import metrics
from io_utils import (ImageSeriesError, append_metrics, read_gray,
                      read_series, save_image)
from processing import (ALIGN_METHODS, COMBINE_METHODS, ContourNotFoundError,
                        SeriesResult, mask_iou, process_series)


def positive_float(text: str) -> float:
    """Тип аргумента argparse: число больше нуля.

    Parameters
    ----------
    text : str
        Строка из командной строки.

    Returns
    -------
    float
        Число.
    """
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' - не число")
    if value <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return value


def threshold_value(text: str) -> int:
    """Тип аргумента argparse: порог от 0 до 254 (0 - метод Оцу).

    Parameters
    ----------
    text : str
        Строка из командной строки.

    Returns
    -------
    int
        Порог.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' - не целое число")
    if not 0 <= value <= 254:
        raise argparse.ArgumentTypeError("порог должен быть от 0 до 254")
    return value


def positive_int(text: str) -> int:
    """Тип аргумента argparse: целое число не меньше 1.

    Parameters
    ----------
    text : str
        Строка из командной строки.

    Returns
    -------
    int
        Число.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' - не целое число")
    if value < 1:
        raise argparse.ArgumentTypeError("значение должно быть не меньше 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Создаёт разборщик аргументов командной строки.

    Returns
    -------
    argparse.ArgumentParser
        Настроенный разборщик.
    """
    parser = argparse.ArgumentParser(
        description="Выравнивание серии кадров по контуру объекта и усреднение для подавления шума")
    parser.add_argument("--input", required=True, help="каталог с кадрами")
    parser.add_argument("--output", default="results", help="каталог для результатов")
    parser.add_argument("--method", choices=ALIGN_METHODS, default="rigid",
                        help="выравнивание: none, translate (сдвиг), rigid (сдвиг и поворот)")
    parser.add_argument("--combine", choices=COMBINE_METHODS, default="mean",
                        help="объединение кадров: mean или median")
    parser.add_argument("--sigma", type=positive_float, default=2.0,
                        help="сигма размытия перед поиском контура")
    parser.add_argument("--threshold", type=threshold_value, default=0,
                        help="порог бинаризации (0 - метод Оцу)")
    parser.add_argument("--max-frames", type=positive_int, default=None,
                        help="использовать только первые N кадров")
    parser.add_argument("--reference", default=None,
                        help="эталон без шума для MSE/PSNR/SSIM (необязательно)")
    return parser


def draw_overlay(gray: np.ndarray, result: SeriesResult, index: int) -> np.ndarray:
    """Рисует на кадре контур, центр масс и главную ось объекта.

    Parameters
    ----------
    gray : numpy.ndarray
        Кадр в оттенках серого.
    result : SeriesResult
        Результат обработки серии.
    index : int
        Номер кадра.

    Returns
    -------
    numpy.ndarray
        Цветное изображение (BGR) с разметкой.
    """
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image, [result.contours[index]], -1, (0, 0, 255), 2)
    feat = result.features[index]
    center = (int(round(feat.cx)), int(round(feat.cy)))
    cv2.circle(image, center, 4, (0, 255, 0), -1)
    dx, dy = 40 * np.cos(feat.angle), 40 * np.sin(feat.angle)
    cv2.line(image, (int(center[0] - dx), int(center[1] - dy)),
             (int(center[0] + dx), int(center[1] + dy)), (0, 255, 0), 2)
    return image


def save_stages(out: Path, names: list, frames: list, result: SeriesResult) -> None:
    """Сохраняет промежуточные результаты каждого кадра в отдельные файлы.

    Имена файлов формируются из имени кадра и названия этапа.

    Parameters
    ----------
    out : Path
        Каталог результатов.
    names : list of str
        Имена исходных файлов без расширения.
    frames : list of numpy.ndarray
        Исходные кадры.
    result : SeriesResult
        Результат обработки.
    """
    for i, name in enumerate(names):
        save_image(out / "original" / f"{name}_original.png", frames[i])
        save_image(out / "masks" / f"{name}_mask.png", result.masks[i])
        save_image(out / "contours" / f"{name}_contour.jpg", draw_overlay(frames[i], result, i))
        save_image(out / "aligned" / f"{name}_aligned.png", result.aligned[i])


def make_row(series: str, args: argparse.Namespace, label: str, image: np.ndarray,
             reference: np.ndarray | None, object_mask: np.ndarray,
             n_frames: int, seconds: float, iou: float | None) -> dict:
    """Собирает строку CSV с метриками для одного изображения.

    Parameters
    ----------
    series : str
        Имя серии (каталога).
    args : argparse.Namespace
        Параметры запуска.
    label : str
        Название метода для колонки method.
    image : numpy.ndarray
        Оцениваемое изображение.
    reference : numpy.ndarray or None
        Эталон без шума (если есть).
    object_mask : numpy.ndarray
        Маска объекта (нужна для оценки шума на фоне).
    n_frames : int
        Сколько кадров использовано.
    seconds : float
        Время работы.
    iou : float or None
        Среднее IoU масок после выравнивания.

    Returns
    -------
    dict
        Строка для append_metrics.
    """
    row = {"series": series, "method": label, "combine": args.combine,
           "sigma": args.sigma, "threshold": args.threshold, "frames": n_frames,
           "noise_std": round(metrics.noise_std(image, object_mask), 3),
           "mean_iou": "" if iou is None else round(iou, 4),
           "time_s": round(seconds, 4)}
    if reference is not None:
        row["mse"] = round(metrics.mse(image, reference), 3)
        row["psnr"] = round(metrics.psnr(image, reference), 3)
        row["ssim"] = round(metrics.ssim(image, reference), 4)
    return row


def run(args: argparse.Namespace) -> None:
    """Выполняет обработку серии и сохраняет все результаты.

    Parameters
    ----------
    args : argparse.Namespace
        Параметры запуска.
    """
    names, frames = read_series(args.input, args.max_frames)
    if len(frames) < 2:
        raise ImageSeriesError("Для усреднения нужно минимум два кадра")
    reference = read_gray(args.reference) if args.reference else None
    if reference is not None and reference.shape != frames[0].shape:
        raise ImageSeriesError("Размер эталона не совпадает с размером кадров")

    start = time.perf_counter()
    result = process_series(frames, args.method, args.combine, args.sigma, args.threshold)
    seconds = time.perf_counter() - start

    out = Path(args.output)
    save_stages(out, names, frames, result)
    final_name = f"result_{args.method}_{args.combine}.png"
    save_image(out / final_name, result.result)

    ious = [mask_iou(result.masks[0], m) for m in result.aligned_masks]
    mean_iou = float(np.mean(ious))
    csv_path = out / "metrics.csv"
    series = Path(args.input).name
    ref_mask = result.masks[0]
    append_metrics(csv_path, make_row(series, args, "single_frame", frames[0], reference,
                                      ref_mask, 1, 0.0, None))
    append_metrics(csv_path, make_row(series, args, args.method, result.result, reference,
                                      ref_mask, len(frames), seconds, mean_iou))

    print(f"Обработано кадров: {len(frames)}")
    print(f"Метод выравнивания: {args.method}, объединение: {args.combine}")
    print(f"Среднее IoU масок после выравнивания: {mean_iou:.3f}")
    print(f"Шум на фоне: {metrics.noise_std(frames[0], ref_mask):.2f} (один кадр) -> "
          f"{metrics.noise_std(result.result, ref_mask):.2f} (результат)")
    if reference is not None:
        print(f"PSNR относительно эталона: {metrics.psnr(frames[0], reference):.2f} дБ -> "
              f"{metrics.psnr(result.result, reference):.2f} дБ")
    print(f"Результат: {out / final_name}")
    print(f"Метрики: {csv_path}")


def main(argv: list | None = None) -> int:
    """Запускает программу.

    Parameters
    ----------
    argv : list, optional
        Аргументы командной строки (по умолчанию sys.argv).

    Returns
    -------
    int
        Код возврата: 0 - успех, 1 - ошибка обработки (argparse при неверных
        аргументах завершает программу с кодом 2).
    """
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ImageSeriesError, ContourNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
