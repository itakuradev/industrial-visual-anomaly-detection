"""OpenCVによる画像前処理Pipeline。

処理順序はここで固定し（crop → resize → denoise → contrast → CLAHE → normalize）、
各処理のON/OFFとパラメータは設定ファイルで切り替える。

- 入力・出力はRGB（uint8, HxWx3）。元の配列は変更しない
- ``normalize`` は画素値を0〜255へ引き伸ばすMin-Max正規化。
  Backbone用のImageNet正規化はAnomalibのPreProcessor側で常に行う
- GT Maskには幾何変換（crop / resize）のみを同じ条件で適用する
"""

from __future__ import annotations

import cv2
import numpy as np
from matplotlib.figure import Figure

from anomaly_detection.utils.config import PreprocessingConfig


class ImagePreprocessor:
    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self._clahe = (
            cv2.createCLAHE(
                clipLimit=config.clahe.clip_limit,
                tileGridSize=(config.clahe.tile_grid_size, config.clahe.tile_grid_size),
            )
            if config.clahe.enabled
            else None
        )

    @property
    def output_size(self) -> tuple[int, int]:
        """Modelへ入力する画像サイズ（height, width）。"""
        return self.config.resize.height, self.config.resize.width

    def enabled_steps(self) -> list[str]:
        """有効な処理名を処理順に返す。"""
        c = self.config
        steps = {
            "crop": c.crop.enabled,
            "resize": c.resize.enabled,
            "denoise": c.denoise.enabled,
            "contrast": c.contrast.enabled,
            "clahe": c.clahe.enabled,
            "normalize": c.normalize.enabled,
        }
        return [name for name, enabled in steps.items() if enabled]

    def summary(self) -> str:
        """Experiment Summary用の短い説明（例: ``resize256x256+clahe``）。"""
        parts = []
        for step in self.enabled_steps():
            if step == "resize":
                parts.append(f"resize{self.config.resize.width}x{self.config.resize.height}")
            elif step == "denoise":
                parts.append(f"denoise-{self.config.denoise.method}")
            else:
                parts.append(step)
        return "+".join(parts) or "none"

    def process(self, image: np.ndarray) -> np.ndarray:
        """RGB画像に前処理を適用した新しい配列を返す。"""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"前処理の入力はRGBのuint8画像（HxWx3）である必要があります"
                f"（dtype={image.dtype}, shape={image.shape}）。"
            )
        c = self.config
        out = self._geometry(image, interpolation=None)
        if c.denoise.enabled:
            k = c.denoise.kernel_size
            if c.denoise.method == "gaussian":
                out = cv2.GaussianBlur(out, (k, k), 0)
            else:
                out = cv2.medianBlur(out, k)
        if c.contrast.enabled:
            out = cv2.convertScaleAbs(out, alpha=c.contrast.alpha, beta=c.contrast.beta)
        if self._clahe is not None:
            lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
            lab[..., 0] = self._clahe.apply(lab[..., 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        if c.normalize.enabled:
            out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
        return out.copy() if np.shares_memory(out, image) else out

    def process_mask(self, mask: np.ndarray) -> np.ndarray:
        """GT Maskに画像と同じ幾何変換（crop / resize）を適用する。"""
        return self._geometry(mask, interpolation=cv2.INTER_NEAREST)

    def apply_geometry(self, image: np.ndarray) -> np.ndarray:
        """幾何変換（crop / resize）のみを適用する。

        Anomaly Mapと位置を合わせて元画像を表示するために使う。
        """
        out = self._geometry(image, interpolation=None)
        return out.copy() if np.shares_memory(out, image) else out

    def _geometry(self, image: np.ndarray, interpolation: int | None) -> np.ndarray:
        c = self.config
        out = image
        if c.crop.enabled:
            height, width = image.shape[:2]
            x, y, w, h = c.crop.x, c.crop.y, c.crop.width, c.crop.height
            if x + w > width or y + h > height:
                raise ValueError(
                    f"crop範囲 (x={x}, y={y}, width={w}, height={h}) が画像サイズ "
                    f"(width={width}, height={height}) を超えています。"
                    "preprocessing.crop を確認してください。"
                )
            out = out[y : y + h, x : x + w]
        if c.resize.enabled:
            size = (c.resize.width, c.resize.height)
            if interpolation is None:  # 縮小はINTER_AREA、拡大はINTER_LINEAR
                shrinking = size[0] * size[1] < out.shape[0] * out.shape[1]
                interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
            out = cv2.resize(out, size, interpolation=interpolation)
        return out


def compare_images(
    original: np.ndarray,
    processed: np.ndarray,
    titles: tuple[str, str] = ("Original", "Preprocessed"),
) -> Figure:
    """処理前・処理後の画像を横並びで表示するFigureを返す。"""
    figure = Figure(figsize=(10, 5))
    for ax, image, title in zip(figure.subplots(1, 2), (original, processed), titles, strict=True):
        ax.imshow(image)
        ax.set_title(f"{title} {image.shape[1]}x{image.shape[0]}")
        ax.axis("off")
    figure.tight_layout()
    return figure
