"""Генератор синтетической серии кадров: объект со сдвигами, поворотами и шумом."""

import argparse
import sys

import cv2
import numpy as np

from io_utils import save_image, ImageSeriesError


def make_clean(height: int = 300, width: int = 400) -> np.ndarray:
    """Рисует эталонное (чистое) изображение объекта.

    Объект несимметричный (L-образная фигура с деталями), чтобы его
    ориентацию можно было определить однозначно.

    Parameters
    ----------
    height, width : int, optional
        Размер изображения.

    Returns
    -------
    numpy.ndarray
        Изображение uint8: яркий объект на тёмном фоне.
    """
    image = np.full((height, width), 60, np.uint8)  # однородный фон
    body = np.array([[130, 70], [200, 70], [200, 170], [290, 170],
                     [290, 230], [130, 230]], np.int32)
    cv2.fillPoly(image, [body], 200)
    cv2.circle(image, (165, 110), 18, 120, -1)           # тёмное пятно
    cv2.rectangle(image, (230, 190), (270, 215), 90, -1)  # тёмная вставка
    cv2.line(image, (140, 210), (190, 210), 140, 3)       # светлая риска
    return image


def make_series(n: int = 16, noise_sigma: float = 25.0, impulse: float = 0.01,
                max_shift: float = 12.0, max_angle: float = 4.0, seed: int = 42):
    """Создаёт серию зашумлённых кадров со случайными смещениями объекта.

    Первый кадр не смещён и совпадает с эталоном по положению.

    Parameters
    ----------
    n : int, optional
        Число кадров.
    noise_sigma : float, optional
        Сигма гауссова шума.
    impulse : float, optional
        Доля пикселей с импульсным шумом («соль и перец»).
    max_shift : float, optional
        Максимальный сдвиг в пикселях.
    max_angle : float, optional
        Максимальный поворот в градусах.
    seed : int, optional
        Начальное значение генератора случайных чисел (для воспроизводимости).

    Returns
    -------
    clean : numpy.ndarray
        Эталонное изображение без шума.
    frames : list of numpy.ndarray
        Зашумлённые кадры.
    params : list of tuple
        Настоящие параметры (dx, dy, угол) каждого кадра.
    """
    rng = np.random.default_rng(seed)
    clean = make_clean()
    h, w = clean.shape
    frames, params = [], []
    for i in range(n):
        if i == 0:
            dx = dy = angle = 0.0
        else:
            dx, dy = rng.uniform(-max_shift, max_shift, 2)
            angle = rng.uniform(-max_angle, max_angle)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        matrix[:, 2] += (dx, dy)
        moved = cv2.warpAffine(clean, matrix, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        noisy = moved.astype(np.float32) + rng.normal(0, noise_sigma, moved.shape)
        if impulse > 0:
            r = rng.random(moved.shape)
            noisy[r < impulse / 2] = 0
            noisy[r > 1 - impulse / 2] = 255
        frames.append(np.clip(noisy, 0, 255).astype(np.uint8))
        params.append((float(dx), float(dy), float(angle)))
    return clean, frames, params


def main(argv: list | None = None) -> int:
    """Сохраняет сгенерированную серию в каталог.

    Parameters
    ----------
    argv : list, optional
        Аргументы командной строки (по умолчанию sys.argv).

    Returns
    -------
    int
        Код возврата: 0 - успех, 1 - ошибка.
    """
    parser = argparse.ArgumentParser(description="Генерация тестовой серии кадров")
    parser.add_argument("--output", default="images", help="каталог для кадров")
    parser.add_argument("--reference", default="reference/clean.png",
                        help="куда сохранить эталон без шума")
    parser.add_argument("--frames", type=int, default=16, help="число кадров")
    parser.add_argument("--noise", type=float, default=25.0, help="сигма гауссова шума")
    parser.add_argument("--impulse", type=float, default=0.01, help="доля импульсного шума")
    parser.add_argument("--seed", type=int, default=42, help="seed генератора")
    args = parser.parse_args(argv)
    if args.frames < 1 or args.noise < 0 or not 0 <= args.impulse < 1:
        print("Ошибка: недопустимое значение параметра", file=sys.stderr)
        return 2
    clean, frames, _ = make_series(args.frames, args.noise, args.impulse, seed=args.seed)
    try:
        for i, frame in enumerate(frames):
            save_image(f"{args.output}/frame_{i:02d}.png", frame)
        save_image(args.reference, clean)
    except ImageSeriesError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    print(f"Сохранено кадров: {len(frames)} в {args.output}, эталон: {args.reference}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
