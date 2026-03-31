"""Generate sample test images with text for OCR testing."""

import numpy as np

try:
    import cv2
except ImportError:
    print("[!] opencv-python not installed. Run: pip install opencv-python")
    raise SystemExit(1)

from pathlib import Path


def create_english_sample(output_dir: str = "samples") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    img = np.ones((400, 800, 3), dtype=np.uint8) * 255

    cv2.putText(img, "Hello, World!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
    cv2.putText(img, "Multilingual OCR Extractor", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "OpenCV + EasyOCR", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 2)
    cv2.putText(img, "Line 4: Testing 12345", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Line 5: Special chars @#$%", (50, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    path = str(Path(output_dir) / "sample_english.png")
    cv2.imwrite(path, img)
    print(f"[*] Created: {path}")
    return path


def create_noisy_sample(output_dir: str = "samples") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    img = np.ones((300, 700, 3), dtype=np.uint8) * 240

    noise = np.random.randint(0, 40, img.shape, dtype=np.uint8)
    img = cv2.subtract(img, noise)

    cv2.putText(img, "Noisy Document Text", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 2)
    cv2.putText(img, "Testing preprocessing", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 2)
    cv2.putText(img, "Denoise + Threshold", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 60), 2)

    path = str(Path(output_dir) / "sample_noisy.png")
    cv2.imwrite(path, img)
    print(f"[*] Created: {path}")
    return path


if __name__ == "__main__":
    create_english_sample()
    create_noisy_sample()
    print("[*] Sample images created in samples/")
