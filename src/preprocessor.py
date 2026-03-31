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

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def remove_colored_lines(self, image: np.ndarray) -> np.ndarray:
        """Remove blue/red ruled lines from notebook paper using color masking in HSV."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        blue_lower, blue_upper = np.array([90, 30, 100]), np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        red_lower1, red_upper1 = np.array([0, 30, 100]), np.array([10, 255, 255])
        red_lower2, red_upper2 = np.array([160, 30, 100]), np.array([180, 255, 255])
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, red_lower1, red_upper1),
            cv2.inRange(hsv, red_lower2, red_upper2),
        )

        lines_mask = cv2.bitwise_or(blue_mask, red_mask)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(image.shape[1] // 8, 40), 1))
        lines_mask = cv2.morphologyEx(lines_mask, cv2.MORPH_CLOSE, horizontal_kernel)
        lines_mask = cv2.dilate(lines_mask, np.ones((3, 1), np.uint8), iterations=1)

        result = image.copy()
        result[lines_mask > 0] = [255, 255, 255]

        repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray_result = cv2.morphologyEx(gray_result, cv2.MORPH_CLOSE, repair_kernel)
        return gray_result

    def remove_ruled_lines_morph(self, image: np.ndarray) -> np.ndarray:
        """Remove horizontal ruled lines using morphological operations (grayscale input)."""
        inverted = cv2.bitwise_not(image)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1] // 4, 1))
        lines_only = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        cleaned = cv2.subtract(inverted, lines_only)
        result = cv2.bitwise_not(cleaned)
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, repair_kernel)
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

    def _has_colored_lines(self, image: np.ndarray) -> bool:
        """Detect if an image contains colored ruled lines (blue/red notebook lines)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        high_sat_ratio = np.mean(saturation > 40)
        return high_sat_ratio > 0.06

    def process_handwritten(self, image_path: str) -> np.ndarray | None:
        """Optimized for handwritten text on lined/plain paper.

        Adapts based on the image content:
        - If colored ruled lines are detected (blue/red notebook lines), uses
          per-pixel RGB minimum to suppress them while keeping dark ink.
        - If no colored lines, returns None to signal that the raw file path
          should be passed directly to EasyOCR (its internal PIL-based loading
          preserves more detail for thin cursive strokes).
        """
        image = self.load_image(image_path)
        image = self.resize_image(image)

        if len(image.shape) == 3 and image.shape[2] == 3 and self._has_colored_lines(image):
            gray = np.min(image, axis=2)
            return cv2.GaussianBlur(gray, (3, 3), 0)

        return None
