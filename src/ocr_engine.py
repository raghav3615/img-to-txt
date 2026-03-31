import easyocr
import numpy as np
from typing import Optional


class OCREngine:
    """Multilingual OCR engine powered by EasyOCR."""

    DEFAULT_LANGUAGES = ["en"]

    def __init__(self, languages: Optional[list[str]] = None, gpu: bool = False):
        self.languages = languages or self.DEFAULT_LANGUAGES
        self.gpu = gpu
        self._reader = None

    @property
    def reader(self) -> easyocr.Reader:
        if self._reader is None:
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def extract_text(self, image: np.ndarray, detail: bool = False) -> list:
        if detail:
            return self._extract_detailed(image)
        return self._extract_plain(image)

    def _extract_plain(self, image: np.ndarray) -> list[str]:
        results = self.reader.readtext(image)
        return [entry[1] for entry in results]

    def _extract_detailed(self, image: np.ndarray) -> list[dict]:
        results = self.reader.readtext(image)
        detailed = []
        for bbox, text, confidence in results:
            detailed.append({
                "text": text,
                "confidence": round(float(confidence), 4),
                "bounding_box": {
                    "top_left": [int(bbox[0][0]), int(bbox[0][1])],
                    "top_right": [int(bbox[1][0]), int(bbox[1][1])],
                    "bottom_right": [int(bbox[2][0]), int(bbox[2][1])],
                    "bottom_left": [int(bbox[3][0]), int(bbox[3][1])],
                },
            })
        return detailed

    def extract_from_file(
        self, image_path: str, detail: bool = False
    ) -> list:
        """Directly read and extract text from an image file without preprocessing."""
        results = self.reader.readtext(image_path)
        if detail:
            return self._format_detailed(results)
        return [entry[1] for entry in results]

    def _format_detailed(self, results: list) -> list[dict]:
        detailed = []
        for bbox, text, confidence in results:
            detailed.append({
                "text": text,
                "confidence": round(float(confidence), 4),
                "bounding_box": {
                    "top_left": [int(bbox[0][0]), int(bbox[0][1])],
                    "top_right": [int(bbox[1][0]), int(bbox[1][1])],
                    "bottom_right": [int(bbox[2][0]), int(bbox[2][1])],
                    "bottom_left": [int(bbox[3][0]), int(bbox[3][1])],
                },
            })
        return detailed


def list_supported_languages() -> list[str]:
    """Return all language codes supported by EasyOCR."""
    return [
        "ab", "af", "ar", "as", "az", "be", "bg", "bh", "bn", "bs",
        "ch_sim", "ch_tra", "cs", "cy", "da", "de", "en", "es", "et",
        "fa", "fr", "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr",
        "hu", "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn",
        "ko", "ku", "ky", "la", "lt", "lv", "mg", "mi", "mk", "ml",
        "mn", "mr", "ms", "mt", "my", "ne", "nl", "no", "oc", "or",
        "pa", "pl", "pt", "ro", "ru", "rs_cyrillic", "rs_latin", "sk",
        "sl", "sq", "sv", "sw", "ta", "te", "tg", "th", "tl", "tr",
        "ug", "uk", "ur", "uz", "vi",
    ]
