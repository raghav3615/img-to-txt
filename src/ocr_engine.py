import easyocr
import numpy as np
from typing import Optional


class OCREngine:
    DEFAULT_LANGUAGES = ["en", "hi"]

    def __init__(self, languages: Optional[list[str]] = None, gpu: bool = False):
        self.languages = languages or self.DEFAULT_LANGUAGES
        self.gpu = gpu
        self._reader = None

    @property
    def reader(self) -> easyocr.Reader:
        if self._reader is None:
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def extract_text(self, image: np.ndarray, detail: bool = False, handwritten: bool = False) -> list:
        if detail:
            return self._extract_detailed(image, handwritten)
        return self._extract_plain(image, handwritten)

    def extract_from_file(
        self, image_path: str, detail: bool = False, handwritten: bool = False
    ) -> list:
        kwargs = self._ocr_kwargs(handwritten)
        results = self.reader.readtext(image_path, **kwargs)
        if detail:
            if handwritten and results:
                return self._detailed_from_lines(results)
            return self._format_detailed(results)
        if handwritten and results:
            lines = self._group_into_lines(results)
            return [" ".join(e[4] for e in line) for line in lines]
        return [entry[1] for entry in results]

    def _ocr_kwargs(self, handwritten: bool) -> dict:
        if handwritten:
            return {
                "paragraph": False,
                "width_ths": 1.5,
                "ycenter_ths": 0.5,
                "height_ths": 0.8,
                "contrast_ths": 0.05,
                "adjust_contrast": 0.7,
            }
        return {}

    def _group_into_lines(self, results: list) -> list:
        """Merge text boxes that share the same vertical band into lines.

        Uses y-center proximity relative to box height as the grouping
        threshold, then sorts each line left-to-right.
        """
        if not results:
            return results

        entries = []
        for bbox, text, confidence in results:
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            x_left = bbox[0][0]
            box_height = abs(bbox[2][1] - bbox[0][1])
            entries.append((y_center, x_left, box_height, bbox, text, confidence))

        entries.sort(key=lambda e: (e[0], e[1]))

        lines = []
        current_line = [entries[0]]

        for entry in entries[1:]:
            prev_y = np.mean([e[0] for e in current_line])
            avg_height = np.mean([e[2] for e in current_line])
            threshold = max(avg_height * 0.35, 10)

            if abs(entry[0] - prev_y) <= threshold:
                current_line.append(entry)
            else:
                lines.append(current_line)
                current_line = [entry]
        lines.append(current_line)

        for line in lines:
            line.sort(key=lambda e: e[1])
        return lines

    def _extract_plain(self, image: np.ndarray, handwritten: bool = False) -> list[str]:
        kwargs = self._ocr_kwargs(handwritten)
        results = self.reader.readtext(image, **kwargs)
        if handwritten and results:
            lines = self._group_into_lines(results)
            return [" ".join(e[4] for e in line) for line in lines]
        return [entry[1] for entry in results]

    def _extract_detailed(self, image: np.ndarray, handwritten: bool = False) -> list[dict]:
        kwargs = self._ocr_kwargs(handwritten)
        results = self.reader.readtext(image, **kwargs)
        if handwritten and results:
            return self._detailed_from_lines(results)
        return self._format_detailed(results)

    def _detailed_from_lines(self, results: list) -> list[dict]:
        lines = self._group_into_lines(results)
        detailed = []
        for line in lines:
            text = " ".join(e[4] for e in line)
            avg_conf = np.mean([e[5] for e in line])
            first_bbox = line[0][3]
            last_bbox = line[-1][3]
            detailed.append({
                "text": text,
                "confidence": round(float(avg_conf), 4),
                "bounding_box": {
                    "top_left": [int(first_bbox[0][0]), int(first_bbox[0][1])],
                    "top_right": [int(last_bbox[1][0]), int(last_bbox[1][1])],
                    "bottom_right": [int(last_bbox[2][0]), int(last_bbox[2][1])],
                    "bottom_left": [int(first_bbox[3][0]), int(first_bbox[3][1])],
                },
            })
        return detailed

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
