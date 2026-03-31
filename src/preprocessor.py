import cv2
import numpy as np
from pathlib import Path


class ImagePreprocessor:
    """Applies computer vision preprocessing techniques to improve OCR accuracy."""

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
        image = cv2.imread(str(path))
        if image is None:
            raise IOError(f"Failed to read image: {image_path}")
        return image

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

    def upscale_image(self, image: np.ndarray, min_dimension: int = 2000) -> np.ndarray:
        h, w = image.shape[:2]
        if min(h, w) >= min_dimension:
            return image
        scale = min_dimension / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def remove_ruled_lines(self, image: np.ndarray) -> np.ndarray:
        """Remove horizontal ruled lines from notebook/lined paper."""
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1] // 3, 1))
        lines_mask = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        _, lines_binary = cv2.threshold(lines_mask, 127, 255, cv2.THRESH_BINARY)
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        result = cv2.add(image, lines_binary)
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
        return result

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
        """Light preprocessing: grayscale + contrast only. Better for colored/complex documents."""
        image = self.load_image(image_path)
        if self.resize:
            image = self.resize_image(image)
        image = self.to_grayscale(image)
        image = self.enhance_contrast(image)
        return image

    def process_handwritten(self, image_path: str) -> np.ndarray:
        """Optimized for handwritten text on lined/plain paper.

        Preserves stroke features while cleaning the background:
        grayscale -> bilateral filter (edge-preserving smooth) -> CLAHE contrast.
        Avoids binarization and heavy denoising to keep stroke texture intact
        for the OCR model's neural network.
        """
        image = self.load_image(image_path)
        image = self.resize_image(image)
        gray = self.to_grayscale(image)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        return enhanced
