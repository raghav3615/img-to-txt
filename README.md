# Multilingual OCR Image-to-Text Extractor

A computer vision project that extracts text from images using **OpenCV** for image preprocessing and **EasyOCR** for multilingual optical character recognition. Supports **80+ languages** and provides a fully command-line driven interface.

## Features

- **Multilingual OCR**: Extract text in 80+ languages including English, Hindi, Chinese, Arabic, Japanese, Korean, French, German, and more
- **Image Preprocessing Pipeline**: Grayscale conversion, adaptive thresholding, noise removal, skew correction, and contrast enhancement using OpenCV
- **Multiple Output Formats**: Save results as plain text, JSON (with bounding boxes and confidence scores), or CSV
- **Batch Processing**: Process all images in a directory with a single command
- **Handwritten Text Support**: Dedicated preprocessing pipeline with bilateral filtering and line-grouping for handwritten documents
- **Configurable Preprocessing**: Choose between full, light, or no preprocessing depending on image quality
- **GPU Support**: Optional CUDA acceleration for faster OCR on large batches

## Project Structure

```
├── main.py                 # CLI entry point
├── src/
│   ├── __init__.py
│   ├── preprocessor.py     # OpenCV image preprocessing pipeline
│   ├── ocr_engine.py       # EasyOCR multilingual text extraction
│   └── utils.py            # Output formatting and file I/O
├── create_sample.py        # Generate test images
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
- `easyocr` — multilingual OCR engine
- `numpy` — array operations for image manipulation
- `Pillow` — additional image format support
- `torch`, `torchvision` — deep learning backend for EasyOCR

### 4. Generate Sample Test Images (Optional)

```bash
python create_sample.py
```

This creates `samples/sample_english.png` and `samples/sample_noisy.png` for testing.

## Usage

### Basic Text Extraction

```bash
python main.py path/to/image.png
```

### Multilingual OCR

Extract text in multiple languages simultaneously:

```bash
python main.py image.jpg -l en hi
python main.py image.png -l ch_sim en
python main.py image.png -l ar en fr
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

### Handwritten Text

For handwritten documents (notebooks, letters, forms), use the `--handwritten` flag which applies a softer preprocessing pipeline and groups detected text into proper lines:

```bash
python main.py handwritten_note.png -l hi en --handwritten
python main.py letter.jpg -l en --handwritten --format json --detail
```

### Preprocessing Options

```bash
python main.py image.png --preprocess full    # Full pipeline (default)
python main.py image.png --preprocess light   # Grayscale + contrast only
python main.py image.png --preprocess none    # No preprocessing
```

### Batch Processing

Process all images in a directory:

```bash
python main.py --batch images_folder/ -l en hi --format json --detail
```

### List Supported Languages

```bash
python main.py --languages
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
| `-l`, `--lang` | Language codes (e.g., `en hi fr`) |
| `-o`, `--output` | Output file path |
| `--format` | Output format: `txt`, `json`, `csv` |
| `--detail` | Include bounding boxes and confidence |
| `--preprocess` | Preprocessing level: `full`, `light`, `none` |
| `--handwritten` | Optimize for handwritten text |
| `--gpu` | Enable CUDA GPU acceleration |
| `--batch` | Process all images in a directory |
| `--languages` | List supported language codes |
| `--no-save` | Print to console only |

## Image Preprocessing Pipeline

### Standard Mode (`--preprocess full`)

Applies the following computer vision techniques for printed text:

1. **Resizing** — Scales large images to a maximum dimension to optimize processing
2. **Grayscale Conversion** — Converts color images to single-channel grayscale
3. **CLAHE Contrast Enhancement** — Applies Contrast Limited Adaptive Histogram Equalization for improved text visibility
4. **Denoising** — Non-local means denoising to remove image noise
5. **Skew Correction** — Detects and corrects rotated text using minimum area rectangle fitting
6. **Adaptive Thresholding** — Gaussian adaptive threshold for binarization, separating text from background

### Handwritten Mode (`--handwritten`)

Uses a softer pipeline designed to preserve pen stroke features:

1. **Resizing** — Limits maximum dimension
2. **Grayscale Conversion** — Single-channel conversion
3. **Bilateral Filtering** — Edge-preserving smoothing that removes paper texture noise while keeping ink strokes sharp
4. **CLAHE Contrast Enhancement** — Higher clip limit (3.0) to boost handwritten ink visibility
5. **Line Grouping** — Post-OCR spatial analysis groups detected text boxes into proper lines by vertical position

## Supported Languages (Subset)

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `hi` | Hindi |
| `ch_sim` | Chinese (Simplified) | `ja` | Japanese |
| `ko` | Korean | `ar` | Arabic |
| `fr` | French | `de` | German |
| `es` | Spanish | `ru` | Russian |
| `pt` | Portuguese | `it` | Italian |
| `ta` | Tamil | `te` | Telugu |
| `bn` | Bengali | `mr` | Marathi |

Run `python main.py --languages` for the full list.

## Example Output

### Plain Text Output
```
Hello, World!
Multilingual OCR Extractor
OpenCV + EasyOCR
```

### JSON Detailed Output
```json
[
  {
    "text": "Hello, World!",
    "confidence": 0.9812,
    "bounding_box": {
      "top_left": [50, 40],
      "top_right": [720, 40],
      "bottom_right": [720, 95],
      "bottom_left": [50, 95]
    }
  }
]
```

## Technical Details

- **OCR Engine**: EasyOCR uses CRAFT (Character Region Awareness for Text Detection) for text detection and a CRNN-based model for text recognition
- **Preprocessing**: OpenCV-based pipeline with configurable stages for handling different image qualities
- **Architecture**: Modular design separating preprocessing, OCR, and output formatting concerns

## License

MIT
