import cv2
import numpy as np
from pathlib import Path


class ImagePreprocessor:
    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

    def __init__(self, grayscale=True, denoise=True, threshold=True, deskew=True, resize=True):
        self.grayscale = grayscale
        self.denoise = denoise
        self.threshold = threshold
        self.deskew = deskew
        self.resize = resize

    def load_image(self, image_path: str) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise IOError(f"Failed to read image: {image_path}")
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = self._flatten_alpha(image)
        return image

    @staticmethod
    def _flatten_alpha(image: np.ndarray) -> np.ndarray:
        """Composite a BGRA image onto a white background."""
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255.0)
        blended = bgr * alpha[:, :, np.newaxis] + white * (1 - alpha[:, :, np.newaxis])
        return blended.astype(np.uint8)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)

    def apply_threshold(self, image: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 5:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.5:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    def resize_image(self, image: np.ndarray, max_dimension: int = 4000) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) <= max_dimension:
            return image
        scale = max_dimension / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def process(self, image_path: str) -> np.ndarray:
        image = self.load_image(image_path)
        if self.resize:
            image = self.resize_image(image)
        if self.grayscale:
            image = self.to_grayscale(image)
            image = self.enhance_contrast(image)
        if self.denoise:
            image = self.remove_noise(image)
        if self.deskew:
            image = self.correct_skew(image)
        if self.threshold:
            image = self.apply_threshold(image)
        return image

    def process_minimal(self, image_path: str) -> np.ndarray:
        image = self.load_image(image_path)
        if self.resize:
            image = self.resize_image(image)
        image = self.to_grayscale(image)
        image = self.enhance_contrast(image)
        return image

    def _has_colored_lines(self, image: np.ndarray) -> bool:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        high_sat_ratio = np.mean(saturation > 40)
        return high_sat_ratio > 0.06

    def process_handwritten(self, image_path: str) -> np.ndarray | None:
        """Adaptive pipeline for handwritten text on lined or plain paper.

        If colored ruled lines are detected, suppresses them via per-pixel RGB
        minimum (colored pixels have at least one high channel; dark ink is low
        in all three). Returns None when no preprocessing is beneficial, so the
        caller can pass the raw file path directly to EasyOCR.
        """
        image = self.load_image(image_path)
        image = self.resize_image(image)

        if len(image.shape) == 3 and image.shape[2] == 3 and self._has_colored_lines(image):
            gray = np.min(image, axis=2)
            return cv2.GaussianBlur(gray, (3, 3), 0)

        return None
