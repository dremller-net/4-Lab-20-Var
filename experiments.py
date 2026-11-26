"""Эксперименты: влияние метода, числа кадров, sigma и порога на качество.

Результаты (CSV и графики для отчёта) сохраняются в results/experiments.
Запуск: python experiments.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # рисуем в файлы, окно не нужно
import matplotlib.pyplot as plt
import numpy as np

import metrics
from generate_data import make_series
from io_utils import append_metrics
from processing import ALIGN_METHODS, COMBINE_METHODS, mask_iou, process_series

OUT = Path("results/experiments")
FRAME_COUNTS = (2, 4, 8, 16)       # сколько кадров усредняем
SIGMAS = (1.0, 2.0, 4.0)           # размытие перед поиском контура
THRESHOLDS = (0, 100, 140)         # порог (0 - метод Оцу)
COLORS = {"none": "tab:red", "translate": "tab:orange", "rigid": "tab:green"}
STYLES = {"mean": "-", "median": "--"}


def evaluate(frames: list, clean: np.ndarray, align: str, combine: str,
             sigma: float, threshold: int):
    """Запускает конвейер и измеряет качество результата.

    Parameters
    ----------
    frames : list of numpy.ndarray
        Кадры серии.
    clean : numpy.ndarray
        Эталон без шума.
    align, combine : str
        Метод выравнивания и способ объединения.
    sigma : float
        Сигма размытия.
    threshold : int
        Порог бинаризации.

    Returns
    -------
    row : dict
        Строка для CSV.
    result : SeriesResult
        Результат конвейера.
    """
    start = time.perf_counter()
    result = process_series(frames, align, combine, sigma, threshold)
    seconds = time.perf_counter() - start
    iou = float(np.mean([mask_iou(result.masks[0], m) for m in result.aligned_masks]))
    row = {"series": "synthetic", "method": align, "combine": combine, "sigma": sigma,
           "threshold": threshold, "frames": len(frames),
           "mse": round(metrics.mse(result.result, clean), 3),
           "psnr": round(metrics.psnr(result.result, clean), 3),
           "ssim": round(metrics.ssim(result.result, clean), 4),
           "noise_std": round(metrics.noise_std(result.result, result.masks[0]), 3),
           "mean_iou": round(iou, 4), "time_s": round(seconds, 4)}
    return row, result


def plot_frames_effect(rows: list) -> None:
    """График: PSNR и SSIM в зависимости от числа кадров для всех методов.

    Parameters
    ----------
    rows : list of dict
        Строки эксперимента 1.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "psnr", "PSNR, дБ"), (axes[1], "ssim", "SSIM")):
        for align in ALIGN_METHODS:
            for combine in COMBINE_METHODS:
                part = [r for r in rows if r["method"] == align and r["combine"] == combine]
                ax.plot([r["frames"] for r in part], [r[key] for r in part],
                        STYLES[combine], marker="o", color=COLORS[align],
                        label=f"{align}, {combine}")
        ax.set_xlabel("Число усредняемых кадров")
        ax.set_ylabel(title)
        ax.set_xscale("log", base=2)
        ax.set_xticks(FRAME_COUNTS, [str(n) for n in FRAME_COUNTS])
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "psnr_ssim_vs_frames.jpg", dpi=130)
    plt.close(fig)


def plot_params_grid(rows: list) -> None:
    """Таблица-картинка: PSNR при разных sigma и порогах.

    Parameters
    ----------
    rows : list of dict
        Строки эксперимента 2.
    """
    grid = np.zeros((len(SIGMAS), len(THRESHOLDS)))
    for i, sigma in enumerate(SIGMAS):
        for j, thr in enumerate(THRESHOLDS):
            grid[i, j] = next(r["psnr"] for r in rows
                              if r["sigma"] == sigma and r["threshold"] == thr)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    image = ax.imshow(grid, cmap="viridis")
    for i in range(len(SIGMAS)):
        for j in range(len(THRESHOLDS)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="white")
    ax.set_xticks(range(len(THRESHOLDS)), ["Оцу" if t == 0 else str(t) for t in THRESHOLDS])
    ax.set_yticks(range(len(SIGMAS)), [str(s) for s in SIGMAS])
    ax.set_xlabel("Порог бинаризации")
    ax.set_ylabel("sigma размытия")
    ax.set_title("PSNR, дБ (rigid + mean, 16 кадров)")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "params_sigma_threshold.jpg", dpi=130)
    plt.close(fig)


def plot_comparison(clean: np.ndarray, frames: list) -> None:
    """Сравнение изображений: эталон, один кадр и разные методы.

    Parameters
    ----------
    clean : numpy.ndarray
        Эталон без шума.
    frames : list of numpy.ndarray
        Кадры серии.
    """
    panels = [("Эталон (без шума)", clean), ("Один кадр", frames[0])]
    for align, combine in (("none", "mean"), ("translate", "mean"),
                           ("rigid", "mean"), ("rigid", "median")):
        res = process_series(frames, align, combine)
        title = f"{align}, {combine}\nPSNR {metrics.psnr(res.result, clean):.1f} дБ"
        panels.append((title, res.result))
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (title, image) in zip(axes.ravel(), panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(OUT / "comparison.jpg", dpi=130)
    plt.close(fig)


def plot_iou(frames: list) -> None:
    """График: IoU масок каждого кадра с опорным до и после выравнивания.

    Parameters
    ----------
    frames : list of numpy.ndarray
        Кадры серии.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.27
    x = np.arange(len(frames))
    for k, align in enumerate(ALIGN_METHODS):
        res = process_series(frames, align, "mean")
        values = [mask_iou(res.masks[0], m) for m in res.aligned_masks]
        ax.bar(x + (k - 1) * width, values, width, color=COLORS[align], label=align)
    ax.set_xlabel("Номер кадра")
    ax.set_ylabel("IoU маски с опорным кадром")
    ax.set_ylim(0.5, 1.02)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "iou_per_frame.jpg", dpi=130)
    plt.close(fig)


def main() -> int:
    """Запускает все эксперименты и сохраняет CSV и графики.

    Returns
    -------
    int
        Код возврата (0 - успех).
    """
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "metrics.csv"
    if csv_path.exists():
        csv_path.unlink()  # начинаем с чистого файла

    clean, frames, _ = make_series(n=16, seed=42)

    # Базовая линия: один зашумлённый кадр
    append_metrics(csv_path, {
        "series": "synthetic", "method": "single_frame", "combine": "-", "frames": 1,
        "mse": round(metrics.mse(frames[0], clean), 3),
        "psnr": round(metrics.psnr(frames[0], clean), 3),
        "ssim": round(metrics.ssim(frames[0], clean), 4)})

    # Эксперимент 1: метод выравнивания x способ объединения x число кадров
    rows_frames = []
    for align in ALIGN_METHODS:
        for combine in COMBINE_METHODS:
            for n in FRAME_COUNTS:
                row, _ = evaluate(frames[:n], clean, align, combine, 2.0, 0)
                rows_frames.append(row)
                append_metrics(csv_path, row)

    # Эксперимент 2: sigma x порог (метод rigid, среднее, 16 кадров)
    rows_params = []
    for sigma in SIGMAS:
        for threshold in THRESHOLDS:
            row, _ = evaluate(frames, clean, "rigid", "mean", sigma, threshold)
            rows_params.append(row)
            append_metrics(csv_path, row)

    plot_frames_effect(rows_frames)
    plot_params_grid(rows_params)
    plot_comparison(clean, frames)
    plot_iou(frames)

    best = max(rows_frames, key=lambda r: r["psnr"])
    print(f"Строк в CSV: {1 + len(rows_frames) + len(rows_params)}")
    print(f"Лучший вариант: {best['method']} + {best['combine']}, "
          f"{best['frames']} кадров, PSNR {best['psnr']} дБ")
    print(f"Файлы сохранены в {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
