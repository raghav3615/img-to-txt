import easyocr
import numpy as np
import re
from typing import Optional


class OCREngine:
    DEFAULT_LANGUAGES = ["en", "hi"]

    def __init__(self, languages: Optional[list[str]] = None, gpu: bool = False):
        self.languages = languages or self.DEFAULT_LANGUAGES
        self.gpu = gpu
        self._readers: dict[tuple[str, ...], easyocr.Reader] = {}

    @property
    def reader(self) -> easyocr.Reader:
        return self._get_reader(self.languages)

    def _get_reader(self, languages: list[str]) -> easyocr.Reader:
        key = tuple(languages)
        if key not in self._readers:
            self._readers[key] = easyocr.Reader(languages, gpu=self.gpu)
        return self._readers[key]

    def extract_text(self, image: np.ndarray, detail: bool = False, handwritten: bool = False) -> list:
        if detail:
            return self._extract_detailed(image, handwritten)
        return self._extract_plain(image, handwritten)

    def extract_from_file(
        self, image_path: str, detail: bool = False, handwritten: bool = False
    ) -> list:
        kwargs = self._ocr_kwargs(handwritten)
        results = self._read_with_adaptive_language(image_path, kwargs)
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
                "decoder": "beamsearch",
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
        results = self._read_with_adaptive_language(image, kwargs)
        if handwritten and results:
            lines = self._group_into_lines(results)
            return [self._normalize_text(" ".join(e[4] for e in line)) for line in lines]
        return [self._normalize_text(entry[1]) for entry in results]

    def _extract_detailed(self, image: np.ndarray, handwritten: bool = False) -> list[dict]:
        kwargs = self._ocr_kwargs(handwritten)
        results = self._read_with_adaptive_language(image, kwargs)
        if handwritten and results:
            return self._detailed_from_lines(results)
        return self._format_detailed(results)

    def _read_with_adaptive_language(self, image_or_path, kwargs: dict) -> list:
        base_results = self.reader.readtext(image_or_path, **kwargs)

        if not self._can_adapt_language() or not base_results:
            return base_results

        dominant = self._dominant_script(base_results)
        if dominant is None:
            return base_results

        alt_results = self._get_reader([dominant]).readtext(image_or_path, **kwargs)
        if not alt_results:
            return base_results

        base_score = self._script_fit_score(base_results, dominant)
        alt_score = self._script_fit_score(alt_results, dominant)
        if alt_score >= base_score + 0.02:
            return alt_results

        return base_results

    def _can_adapt_language(self) -> bool:
        return len(self.languages) == 2 and set(self.languages) == {"en", "hi"}

    @staticmethod
    def _result_text(entry) -> str:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            return str(entry[1])
        return ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _result_confidence(entry) -> float:
        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
            try:
                return float(entry[2])
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def _dominant_script(self, results: list) -> str | None:
        text = " ".join(self._result_text(entry) for entry in results)
        latin_chars = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
        devanagari_chars = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")

        if latin_chars == 0 and devanagari_chars == 0:
            return None
        if latin_chars >= devanagari_chars * 1.35:
            return "en"
        if devanagari_chars >= latin_chars * 1.35:
            return "hi"
        return None

    def _script_fit_score(self, results: list, target_script: str) -> float:
        text = " ".join(self._result_text(entry) for entry in results)
        latin_chars = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
        devanagari_chars = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")

        total_script_chars = latin_chars + devanagari_chars
        if target_script == "en":
            target_chars = latin_chars
        else:
            target_chars = devanagari_chars
        script_ratio = target_chars / max(total_script_chars, 1)

        confidences = [self._result_confidence(entry) for entry in results]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return (avg_conf * 0.75) + (script_ratio * 0.25)

    def _detailed_from_lines(self, results: list) -> list[dict]:
        lines = self._group_into_lines(results)
        detailed = []
        for line in lines:
            text = self._normalize_text(" ".join(e[4] for e in line))
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
                "text": self._normalize_text(text),
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
