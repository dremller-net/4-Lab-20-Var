"""Обработка серии кадров: поиск контура, выравнивание, усреднение."""

import math
from dataclasses import dataclass

import cv2
import numpy as np

ALIGN_METHODS = ("none", "translate", "rigid")
COMBINE_METHODS = ("mean", "median")
MIN_AREA = 200  # контур меньше этой площади (в пикселях) считаем шумом


class ContourNotFoundError(Exception):
    """Контур объекта на кадре не найден."""


@dataclass
class Features:
    """Положение объекта на кадре.

    Attributes
    ----------
    cx, cy : float
        Центр масс контура (в пикселях).
    angle : float
        Ориентация главной оси контура в радианах.
    """

    cx: float
    cy: float
    angle: float


@dataclass
class SeriesResult:
    """Результат обработки серии кадров.

    Attributes
    ----------
    masks : list of numpy.ndarray
        Бинарные маски объекта на исходных кадрах.
    contours : list of numpy.ndarray
        Контуры объекта (формат OpenCV).
    features : list of Features
        Положение объекта на каждом кадре.
    aligned : list of numpy.ndarray
        Кадры после выравнивания.
    aligned_masks : list of numpy.ndarray
        Маски после выравнивания.
    result : numpy.ndarray
        Итоговое изображение (усреднение выровненных кадров).
    """

    masks: list
    contours: list
    features: list
    aligned: list
    aligned_masks: list
    result: np.ndarray


def segment(gray: np.ndarray, sigma: float = 2.0, threshold: int = 0) -> np.ndarray:
    """Выделяет объект на кадре: размытие, порог, морфология.

    Parameters
    ----------
    gray : numpy.ndarray
        Кадр в оттенках серого.
    sigma : float, optional
        Сигма гауссова размытия (подавляет шум перед порогом).
    threshold : int, optional
        Порог 1...254. Если 0, порог подбирается методом Оцу.

    Returns
    -------
    numpy.ndarray
        Бинарная маска (0 или 255).
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    if threshold == 0:
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    # Если объект тёмный на светлом фоне, маска получится инвертированной.
    # Узнаём это по рамке кадра: там должен быть фон (то есть 0 в маске).
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:
        mask = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # убираем одиночные точки
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # закрываем мелкие дырки
    return mask


def find_contour(mask: np.ndarray) -> np.ndarray:
    """Находит внешний контур самого большого объекта на маске.

    Parameters
    ----------
    mask : numpy.ndarray
        Бинарная маска.

    Returns
    -------
    numpy.ndarray
        Контур в формате OpenCV.

    Raises
    ------
    ContourNotFoundError
        Если объектов нет или они слишком маленькие.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ContourNotFoundError("на кадре не найдено ни одного контура")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_AREA:
        raise ContourNotFoundError("найденный контур слишком мал, проверьте порог и sigma")
    return contour


def filled_mask(contour: np.ndarray, shape: tuple) -> np.ndarray:
    """Рисует закрашенную маску объекта по его контуру.

    Parameters
    ----------
    contour : numpy.ndarray
        Контур объекта.
    shape : tuple
        Размер кадра (высота, ширина).

    Returns
    -------
    numpy.ndarray
        Маска 0/255 без дырок внутри объекта.
    """
    mask = np.zeros(shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask


def contour_features(contour: np.ndarray) -> Features:
    """Считает центр масс и ориентацию контура по моментам.

    Parameters
    ----------
    contour : numpy.ndarray
        Контур объекта.

    Returns
    -------
    Features
        Центр и угол главной оси (по формуле через центральные моменты).
    """
    m = cv2.moments(contour)
    if m["m00"] == 0:
        raise ContourNotFoundError("у контура нулевая площадь")
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    angle = 0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])
    return Features(cx, cy, angle)


def angle_difference(a: float, b: float) -> float:
    """Разность углов a - b, приведённая к диапазону (-pi/2, pi/2].

    Ось объекта не имеет направления, поэтому углы определены с точностью до pi.

    Parameters
    ----------
    a, b : float
        Углы в радианах.

    Returns
    -------
    float
        Разность в радианах.
    """
    return (a - b + math.pi / 2) % math.pi - math.pi / 2


def alignment_matrix(feat: Features, ref: Features, method: str) -> np.ndarray:
    """Строит матрицу аффинного преобразования, совмещающую кадр с эталоном.

    Parameters
    ----------
    feat : Features
        Положение объекта на выравниваемом кадре.
    ref : Features
        Положение объекта на опорном кадре.
    method : str
        "none" (без выравнивания), "translate" (только сдвиг)
        или "rigid" (сдвиг и поворот).

    Returns
    -------
    numpy.ndarray
        Матрица 2x3 для cv2.warpAffine.
    """
    if method == "none":
        return np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    dx = ref.cx - feat.cx
    dy = ref.cy - feat.cy
    if method == "translate":
        return np.array([[1, 0, dx], [0, 1, dy]], np.float64)
    if method == "rigid":
        # В координатах изображения ось y направлена вниз, поэтому знак угла
        # для getRotationMatrix2D совпадает с разностью углов осей.
        diff = angle_difference(feat.angle, ref.angle)
        matrix = cv2.getRotationMatrix2D((feat.cx, feat.cy), math.degrees(diff), 1.0)
        matrix[0, 2] += dx
        matrix[1, 2] += dy
        return matrix
    raise ValueError(f"Неизвестный метод выравнивания: {method}")


def warp(image: np.ndarray, matrix: np.ndarray, nearest: bool = False) -> np.ndarray:
    """Применяет аффинное преобразование к кадру или маске.

    Parameters
    ----------
    image : numpy.ndarray
        Кадр или маска.
    matrix : numpy.ndarray
        Матрица 2x3.
    nearest : bool, optional
        True для масок (без сглаживания границ).

    Returns
    -------
    numpy.ndarray
        Преобразованное изображение того же размера.
    """
    h, w = image.shape[:2]
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(image, matrix, (w, h), flags=flags,
                          borderMode=cv2.BORDER_REPLICATE)


def combine_frames(frames: list, how: str = "mean") -> np.ndarray:
    """Объединяет кадры в одно изображение (усреднение или медиана).

    Parameters
    ----------
    frames : list of numpy.ndarray
        Выровненные кадры одного размера.
    how : str, optional
        "mean" (среднее) или "median" (медиана, устойчива к выбросам).

    Returns
    -------
    numpy.ndarray
        Итоговое изображение uint8.
    """
    if not frames:
        raise ValueError("Нет кадров для усреднения")
    stack = np.stack(frames).astype(np.float32)  # в float, чтобы не было переполнения uint8
    if how == "mean":
        result = stack.mean(axis=0)
    elif how == "median":
        result = np.median(stack, axis=0)
    else:
        raise ValueError(f"Неизвестный способ объединения: {how}")
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Считает IoU (пересечение к объединению) двух масок.

    Parameters
    ----------
    a, b : numpy.ndarray
        Бинарные маски одного размера.

    Returns
    -------
    float
        Число от 0 до 1; чем ближе к 1, тем лучше совпадают маски.
    """
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter / union) if union else 0.0


def process_series(frames: list, align: str = "rigid", combine: str = "mean",
                   sigma: float = 2.0, threshold: int = 0) -> SeriesResult:
    """Полный конвейер: контуры, выравнивание по первому кадру, усреднение.

    Parameters
    ----------
    frames : list of numpy.ndarray
        Серия кадров в оттенках серого (первый кадр опорный).
    align : str, optional
        Метод выравнивания: "none", "translate" или "rigid".
    combine : str, optional
        Способ объединения: "mean" или "median".
    sigma : float, optional
        Сигма размытия перед поиском контура.
    threshold : int, optional
        Порог бинаризации (0 - метод Оцу).

    Returns
    -------
    SeriesResult
        Маски, контуры, выровненные кадры и итоговое изображение.
    """
    if align not in ALIGN_METHODS:
        raise ValueError(f"Неизвестный метод выравнивания: {align}")
    if combine not in COMBINE_METHODS:
        raise ValueError(f"Неизвестный способ объединения: {combine}")
    if sigma <= 0:
        raise ValueError("sigma должна быть больше нуля")
    if not 0 <= threshold <= 254:
        raise ValueError("threshold должен быть от 0 до 254")
    if not frames:
        raise ValueError("Нет кадров для обработки")

    masks, contours, features = [], [], []
    for index, frame in enumerate(frames):
        try:
            contour = find_contour(segment(frame, sigma, threshold))
        except ContourNotFoundError as error:
            raise ContourNotFoundError(f"кадр №{index}: {error}")
        contours.append(contour)
        masks.append(filled_mask(contour, frame.shape))
        features.append(contour_features(contour))

    aligned, aligned_masks = [], []
    for frame, mask, feat in zip(frames, masks, features):
        matrix = alignment_matrix(feat, features[0], align)
        aligned.append(warp(frame, matrix))
        aligned_masks.append(warp(mask, matrix, nearest=True))
    result = combine_frames(aligned, combine)
    return SeriesResult(masks, contours, features, aligned, aligned_masks, result)
