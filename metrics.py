"""Метрики качества: MSE, PSNR, SSIM и оценка остаточного шума."""

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Среднеквадратичная ошибка между двумя изображениями.

    Parameters
    ----------
    a, b : numpy.ndarray
        Изображения одного размера.

    Returns
    -------
    float
        Значение MSE (0 для одинаковых изображений).
    """
    if a.shape != b.shape:
        raise ValueError("Изображения должны быть одного размера")
    diff = a.astype(np.float64) - b.astype(np.float64)  # float, иначе uint8 переполнится
    return float(np.mean(diff ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_value: float = 255.0) -> float:
    """Пиковое отношение сигнал/шум в децибелах.

    Parameters
    ----------
    a, b : numpy.ndarray
        Изображения одного размера.
    max_value : float, optional
        Максимально возможная яркость.

    Returns
    -------
    float
        PSNR в дБ (inf для одинаковых изображений).
    """
    error = mse(a, b)
    if error == 0:
        return float("inf")
    return float(10 * np.log10(max_value ** 2 / error))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Индекс структурного сходства двух 8-битных изображений.

    Parameters
    ----------
    a, b : numpy.ndarray
        Изображения одного размера.

    Returns
    -------
    float
        Число от -1 до 1 (1 - полное совпадение).
    """
    return float(structural_similarity(a, b, data_range=255))


def background_mask(object_mask: np.ndarray, margin: int = 15) -> np.ndarray:
    """Маска фона: всё, что дальше margin пикселей от объекта.

    Parameters
    ----------
    object_mask : numpy.ndarray
        Маска объекта (0/255).
    margin : int, optional
        Отступ от объекта в пикселях.

    Returns
    -------
    numpy.ndarray
        Булева маска фона.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    grown = cv2.dilate(object_mask, kernel)
    return grown == 0


def noise_std(image: np.ndarray, object_mask: np.ndarray, margin: int = 15) -> float:
    """Оценивает уровень шума как стандартное отклонение яркости на фоне.

    Работает без эталонного изображения: фон однородный, поэтому его
    разброс яркости - это шум.

    Parameters
    ----------
    image : numpy.ndarray
        Изображение.
    object_mask : numpy.ndarray
        Маска объекта.
    margin : int, optional
        Отступ от объекта.

    Returns
    -------
    float
        Стандартное отклонение яркости фона.
    """
    background = background_mask(object_mask, margin)
    if not background.any():
        raise ValueError("На кадре не осталось фона для оценки шума")
    return float(image[background].astype(np.float64).std())
