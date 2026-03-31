# Multilingual OCR Image-to-Text Extractor

A computer vision project that extracts text from images using **OpenCV** for image preprocessing and **EasyOCR** for optical character recognition. Optimized for **English** and **Hindi** — supports both printed and handwritten text with a fully command-line driven interface.

## Features

- **English & Hindi OCR** — Extracts text in English and Hindi out of the box, no language flags needed
- **Handwritten Text Support** — Adaptive preprocessing that detects and removes colored notebook lines via per-pixel RGB minimum, with spatial line-grouping of detected text boxes
- **Image Preprocessing Pipeline** — Grayscale conversion, adaptive thresholding, noise removal, skew correction, and CLAHE contrast enhancement using OpenCV
- **Transparent PNG Handling** — Automatically composites images with alpha channels onto a white background
- **Multiple Output Formats** — Plain text, JSON (with bounding boxes and confidence scores), or CSV
- **Batch Processing** — Process all images in a directory with a single command
- **Configurable Preprocessing** — Full, light, or no preprocessing depending on image quality
- **GPU Support** — Optional CUDA acceleration for faster OCR

## Project Structure

```
├── main.py                 # CLI entry point
├── src/
│   ├── __init__.py
│   ├── preprocessor.py     # OpenCV image preprocessing pipeline
│   ├── ocr_engine.py       # EasyOCR text extraction engine
│   └── utils.py            # Output formatting and file I/O
├── create_sample.py        # Generate test images with OpenCV
├── samples/                # Sample/test images
├── output/                 # Extracted text output (auto-created)
├── requirements.txt
└── README.md
```

## Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` — image loading and preprocessing
- `easyocr` — OCR engine (English + Hindi models downloaded on first run)
- `numpy` — array operations for image manipulation
- `Pillow` — additional image format support
- `torch`, `torchvision` — deep learning backend for EasyOCR

### 4. Generate Sample Test Images (Optional)

```bash
python create_sample.py
```

Creates `samples/sample_english.png` and `samples/sample_noisy.png` for testing.

## Usage

### Basic Text Extraction

English and Hindi are loaded by default — just pass the image:

```bash
python main.py image.png
python main.py samples/hindi2.png
python main.py samples/sample_english.png
```

### Handwritten Text

For handwritten documents (notebooks, letters, forms), use the `--handwritten` flag:

```bash
python main.py handwritten_note.png --handwritten
python main.py samples/hindi.png --handwritten
```

### Detailed Output (Bounding Boxes + Confidence)

```bash
python main.py image.png --detail
```

### Output Formats

```bash
python main.py image.png --format txt           # Plain text (default)
python main.py image.png --format json --detail  # JSON with coordinates
python main.py image.png --format csv --detail   # CSV with confidence scores
```

### Custom Output Path

```bash
python main.py image.png -o results/extracted_text
```

### Preprocessing Options

```bash
python main.py image.png --preprocess light   # Grayscale + contrast (default)
python main.py image.png --preprocess full    # Full pipeline (denoise, deskew, threshold)
python main.py image.png --preprocess none    # No preprocessing
```

### Batch Processing

Process all images in a directory:

```bash
python main.py --batch images_folder/ --format json --detail
```

### Console-Only Output (No File Saved)

```bash
python main.py image.png --no-save
```

### GPU Acceleration

```bash
python main.py image.png --gpu
```

## Command-Line Reference

| Argument | Description |
|---|---|
| `image` | Path to input image file |
| `-l`, `--lang` | Language codes, default: `en hi` |
| `-o`, `--output` | Output file path |
| `--format` | Output format: `txt`, `json`, `csv` |
| `--detail` | Include bounding boxes and confidence |
| `--preprocess` | Preprocessing level: `light` (default), `full`, `none` |
| `--handwritten` | Optimize for handwritten text |
| `--gpu` | Enable CUDA GPU acceleration |
| `--batch` | Process all images in a directory |
| `--no-save` | Print to console only |
| `--languages` | List all supported language codes |

## Image Preprocessing Pipeline

### Standard Mode (`--preprocess light`, default)

For printed/digital text documents:

1. **Resizing** — Scales large images to a maximum dimension to optimize processing
2. **Grayscale Conversion** — Converts color images to single-channel grayscale
3. **CLAHE Contrast Enhancement** — Contrast Limited Adaptive Histogram Equalization for improved text visibility

### Full Mode (`--preprocess full`)

Adds aggressive cleaning on top of the light pipeline:

4. **Denoising** — Non-local means denoising to remove image noise
5. **Skew Correction** — Detects and corrects rotated text using minimum area rectangle fitting
6. **Adaptive Thresholding** — Gaussian adaptive threshold for binarization

### Handwritten Mode (`--handwritten`)

Adapts based on image content to preserve pen stroke features:

1. **Colored Line Detection** — Analyzes HSV saturation to detect blue/red notebook ruled lines
2. **RGB Minimum Channel** — If colored lines are found, takes the per-pixel minimum across R, G, B channels. Colored lines fade to near-white while dark ink is preserved
3. **Gaussian Smoothing** — Light 3×3 blur to reduce remaining paper texture noise
4. **Raw Passthrough** — If no colored lines are detected, the unmodified image is passed directly to EasyOCR to preserve thin stroke detail
5. **Line Grouping** — Post-OCR spatial analysis groups detected text boxes into proper lines by y-center proximity, then sorts each line left-to-right

## Technical Details

- **OCR Engine**: EasyOCR uses CRAFT (Character Region Awareness for Text Detection) for text detection and a CRNN-based model for recognition
- **Default Languages**: English (`en`) + Hindi (`hi`) loaded together for bilingual recognition
- **Preprocessing**: OpenCV-based pipeline with configurable stages for handling different image qualities
- **Architecture**: Modular design — `preprocessor.py` handles image cleaning, `ocr_engine.py` handles text extraction and line grouping, `utils.py` handles output formatting

## License

MIT

