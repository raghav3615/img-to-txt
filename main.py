import argparse
import sys
import time
from pathlib import Path

from src.preprocessor import ImagePreprocessor
from src.ocr_engine import OCREngine, list_supported_languages
from src.utils import (
    format_results,
    save_output,
    generate_output_filename,
    get_supported_languages,
    print_language_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr-extractor",
        description="Extract text from images with multilingual OCR support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py image.png\n"
            "  python main.py image.jpg -o result.txt\n"
            "  python main.py scan.png --detail --format json\n"
            "  python main.py handwritten.png --handwritten\n"
            "  python main.py document.tiff --preprocess full --format csv -o output/result\n"
            "  python main.py --batch images/\n"
        ),
    )

    parser.add_argument("image", nargs="?", help="Path to input image file")
    parser.add_argument(
        "-l", "--lang",
        nargs="+",
        default=["en", "hi"],
        metavar="CODE",
        help="Language codes for OCR (default: en hi). Use --languages to list all.",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="PATH",
        help="Output file path (auto-generated if not provided)",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "json", "csv"],
        default="txt",
        help="Output format (default: txt). csv/json require --detail.",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Include bounding boxes and confidence scores",
    )
    parser.add_argument(
        "--preprocess",
        choices=["full", "light", "none"],
        default="light",
        help="Preprocessing level: light (grayscale+contrast, default), full (all steps), none",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration (requires CUDA-compatible setup)",
    )
    parser.add_argument(
        "--languages",
        action="store_true",
        help="List all supported language codes and exit",
    )
    parser.add_argument(
        "--batch",
        metavar="DIR",
        help="Process all images in a directory",
    )
    parser.add_argument(
        "--handwritten",
        action="store_true",
        help="Optimize for handwritten text (adaptive preprocessing, line grouping)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print results to console only, do not save to file",
    )

    return parser


def process_single_image(
    image_path: str,
    preprocessor: ImagePreprocessor,
    engine: OCREngine,
    preprocess_mode: str,
    detail: bool,
    handwritten: bool = False,
) -> list:
    if handwritten:
        processed = preprocessor.process_handwritten(image_path)
        if processed is not None:
            return engine.extract_text(processed, detail=detail, handwritten=True)
        return engine.extract_from_file(image_path, detail=detail, handwritten=True)

    if preprocess_mode == "full":
        processed = preprocessor.process(image_path)
    elif preprocess_mode == "light":
        processed = preprocessor.process_minimal(image_path)
    else:
        return engine.extract_from_file(image_path, detail=detail)

    return engine.extract_text(processed, detail=detail)


def run_single(args: argparse.Namespace) -> None:
    print(f"[*] Loading OCR engine for languages: {', '.join(args.lang)}")
    engine = OCREngine(languages=args.lang, gpu=args.gpu)
    preprocessor = ImagePreprocessor()

    mode_label = "handwritten" if args.handwritten else args.preprocess
    print(f"[*] Processing ({mode_label}): {args.image}")
    start = time.time()

    results = process_single_image(
        args.image, preprocessor, engine, args.preprocess, args.detail,
        handwritten=args.handwritten,
    )
    elapsed = time.time() - start

    if not results:
        print("[!] No text detected in the image.")
        return

    use_detail = args.detail or args.format in ("json", "csv")
    if use_detail and not args.detail:
        results = process_single_image(
            args.image, preprocessor, engine, args.preprocess, detail=True,
            handwritten=args.handwritten,
        )

    output_text = format_results(results, detailed=use_detail)

    print(f"\n--- Extracted Text ({elapsed:.2f}s) ---\n")
    print(output_text)

    if not args.no_save:
        out_path = args.output or generate_output_filename(args.image)
        saved = save_output(output_text, out_path, fmt=args.format)
        print(f"\n[*] Saved to: {saved}")


def run_batch(args: argparse.Namespace) -> None:
    batch_dir = Path(args.batch)
    if not batch_dir.is_dir():
        print(f"[!] Directory not found: {args.batch}", file=sys.stderr)
        sys.exit(1)

    image_files = sorted(
        p for p in batch_dir.iterdir()
        if p.suffix.lower() in ImagePreprocessor.SUPPORTED_FORMATS
    )
    if not image_files:
        print(f"[!] No supported images found in: {args.batch}")
        return

    print(f"[*] Found {len(image_files)} images in {args.batch}")
    print(f"[*] Loading OCR engine for languages: {', '.join(args.lang)}")
    engine = OCREngine(languages=args.lang, gpu=args.gpu)
    preprocessor = ImagePreprocessor()

    total_start = time.time()
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_path.name}")
        start = time.time()

        try:
            results = process_single_image(
                str(img_path), preprocessor, engine, args.preprocess, args.detail,
                handwritten=args.handwritten,
            )
        except Exception as e:
            print(f"  [!] Error: {e}")
            continue

        elapsed = time.time() - start

        if not results:
            print(f"  [!] No text detected ({elapsed:.2f}s)")
            continue

        use_detail = args.detail or args.format in ("json", "csv")
        if use_detail and not args.detail:
            results = process_single_image(
                str(img_path), preprocessor, engine, args.preprocess, detail=True,
                handwritten=args.handwritten,
            )

        output_text = format_results(results, detailed=use_detail)
        print(f"  Extracted {len(results)} text regions ({elapsed:.2f}s)")

        if not args.no_save:
            out_path = generate_output_filename(str(img_path))
            saved = save_output(output_text, out_path, fmt=args.format)
            print(f"  Saved to: {saved}")

    total = time.time() - total_start
    print(f"\n[*] Batch complete: {len(image_files)} images in {total:.2f}s")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.languages:
        print_language_table(get_supported_languages())
        print(f"Total supported by EasyOCR: {len(list_supported_languages())} languages")
        sys.exit(0)

    if args.batch:
        run_batch(args)
        return

    if not args.image:
        parser.print_help()
        sys.exit(1)

    if not Path(args.image).exists():
        print(f"[!] File not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    run_single(args)


if __name__ == "__main__":
    main()
