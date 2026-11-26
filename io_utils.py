"""Ввод-вывод: чтение серии изображений и сохранение результатов."""

import csv
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Колонки итогового CSV с результатами измерений
CSV_COLUMNS = ["series", "method", "combine", "sigma", "threshold", "frames",
               "mse", "psnr", "ssim", "noise_std", "mean_iou", "time_s"]


class ImageSeriesError(Exception):
    """Ошибка при чтении, записи или проверке серии изображений."""


def list_images(folder: str | Path) -> list[Path]:
    """Возвращает отсортированный список файлов изображений в каталоге.

    Parameters
    ----------
    folder : str or Path
        Путь к каталогу с кадрами.

    Returns
    -------
    list of Path
        Пути к файлам с поддерживаемым расширением.

    Raises
    ------
    ImageSeriesError
        Если каталога нет или в нём нет изображений.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ImageSeriesError(f"Каталог не найден: {folder}")
    paths = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not paths:
        raise ImageSeriesError(
            f"В каталоге нет изображений ({', '.join(SUPPORTED_EXTENSIONS)}): {folder}")
    return paths


def read_gray(path: str | Path) -> np.ndarray:
    """Читает изображение в оттенках серого.

    Файл читается через numpy и imdecode, потому что cv2.imread
    не умеет открывать пути с русскими буквами в Windows.

    Parameters
    ----------
    path : str or Path
        Путь к файлу.

    Returns
    -------
    numpy.ndarray
        Массив uint8 формы (высота, ширина).

    Raises
    ------
    ImageSeriesError
        Если файл не открывается или не является изображением.
    """
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as error:
        raise ImageSeriesError(f"Не удалось открыть файл {path}: {error}")
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ImageSeriesError(f"Файл повреждён или формат не поддерживается: {path}")
    return image


def read_series(folder: str | Path,
                max_frames: int | None = None) -> tuple[list[str], list[np.ndarray]]:
    """Читает всю серию кадров из каталога.

    Parameters
    ----------
    folder : str or Path
        Каталог с кадрами.
    max_frames : int, optional
        Сколько кадров взять (первые по порядку имён).

    Returns
    -------
    names : list of str
        Имена файлов без расширения.
    frames : list of numpy.ndarray
        Кадры в оттенках серого.

    Raises
    ------
    ImageSeriesError
        Если кадры имеют разный размер или не читаются.
    """
    paths = list_images(folder)
    if max_frames is not None:
        paths = paths[:max_frames]
    frames = [read_gray(p) for p in paths]
    for path, frame in zip(paths, frames):
        if frame.shape != frames[0].shape:
            raise ImageSeriesError(
                f"Размер кадра {path.name} {frame.shape} не совпадает с первым {frames[0].shape}")
    return [p.stem for p in paths], frames


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Сохраняет изображение в файл (создаёт каталоги при необходимости).

    Parameters
    ----------
    path : str or Path
        Куда сохранить. Формат определяется по расширению.
    image : numpy.ndarray
        Изображение uint8.

    Raises
    ------
    ImageSeriesError
        Если запись не удалась.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ImageSeriesError(f"Не удалось закодировать изображение: {path}")
    try:
        buffer.tofile(str(path))
    except OSError as error:
        raise ImageSeriesError(f"Не удалось записать файл {path}: {error}")


def append_metrics(csv_path: str | Path, row: dict) -> None:
    """Добавляет строку с результатами измерений в CSV-файл.

    Parameters
    ----------
    csv_path : str or Path
        Путь к CSV. Если файла нет, он создаётся с заголовком.
    row : dict
        Значения для колонок CSV_COLUMNS (недостающие остаются пустыми).
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
