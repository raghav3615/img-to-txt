import json
import csv
from pathlib import Path
from datetime import datetime


def format_results(texts: list, detailed: bool = False) -> str:
    if detailed:
        return json.dumps(texts, indent=2, ensure_ascii=False)
    return "\n".join(texts)


def save_output(content: str, output_path: str, fmt: str = "txt") -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        path = path.with_suffix(".json")
        path.write_text(content, encoding="utf-8")
    elif fmt == "csv":
        path = path.with_suffix(".csv")
        data = json.loads(content)
        with open(path, "w", newline="", encoding="utf-8") as f:
            if data and isinstance(data[0], dict):
                writer = csv.DictWriter(f, fieldnames=["text", "confidence"])
                writer.writeheader()
                for entry in data:
                    writer.writerow({
                        "text": entry["text"],
                        "confidence": entry["confidence"],
                    })
            else:
                writer = csv.writer(f)
                for line in data:
                    writer.writerow([line])
    else:
        path = path.with_suffix(".txt")
        path.write_text(content, encoding="utf-8")

    return str(path)


def generate_output_filename(image_path: str, output_dir: str = "output") -> str:
    stem = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(output_dir) / f"{stem}_{timestamp}")


def get_supported_languages() -> dict[str, str]:
    return {
        "en": "English", "hi": "Hindi", "mr": "Marathi", "ta": "Tamil",
        "te": "Telugu", "bn": "Bengali", "gu": "Gujarati", "kn": "Kannada",
        "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu", "ar": "Arabic",
        "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
        "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "uk": "Ukrainian",
        "pl": "Polish", "cs": "Czech", "ro": "Romanian", "hu": "Hungarian",
        "sv": "Swedish", "da": "Danish", "no": "Norwegian", "ja": "Japanese",
        "ko": "Korean", "ch_sim": "Chinese (Simplified)", "ch_tra": "Chinese (Traditional)",
        "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay",
        "tl": "Filipino", "tr": "Turkish", "he": "Hebrew", "fa": "Persian",
        "ne": "Nepali", "my": "Myanmar", "km": "Khmer", "la": "Latin",
    }


def print_language_table(languages: dict[str, str]) -> None:
    print(f"\n{'Code':<12} {'Language'}")
    print("-" * 36)
    for code, name in sorted(languages.items(), key=lambda x: x[1]):
        print(f"{code:<12} {name}")
    print()
