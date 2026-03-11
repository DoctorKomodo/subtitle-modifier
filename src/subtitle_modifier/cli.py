"""CLI entry point for subtitle-modifier."""

import argparse
import glob
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="subtitle-modifier",
        description="Convert ALL-UPPERCASE subtitle text to sentence case, preserving proper nouns via NLP.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Input subtitle file(s). Supports glob patterns.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory. If not set, files are saved alongside originals with a '_converted' suffix.",
    )
    parser.add_argument(
        "--model",
        default="en_core_web_sm",
        help="spaCy model to use for NER (default: en_core_web_sm).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files.",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        metavar="MODEL",
        help="Benchmark one or more spaCy models and print a speed comparison table.",
    )

    args = parser.parse_args(argv)

    # Expand glob patterns
    input_files = []
    for pattern in args.files:
        expanded = glob.glob(pattern)
        if not expanded:
            print(f"Warning: no files matched '{pattern}'", file=sys.stderr)
        input_files.extend(expanded)

    if not input_files:
        print("Error: no input files found.", file=sys.stderr)
        sys.exit(1)

    # Benchmark mode
    if args.benchmark:
        from .benchmark import print_results, run_benchmark

        results = run_benchmark(input_files, args.benchmark)
        print_results(results, len(input_files))
        return

    # Lazy-load spaCy after argument parsing (heavy import)
    import spacy

    from .subtitle_io import process_file

    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(
            f"Error: spaCy model '{args.model}' not found. "
            f"Install it with: python -m spacy download {args.model}",
            file=sys.stderr,
        )
        sys.exit(1)

    for filepath in input_files:
        input_p = Path(filepath)

        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / input_p.name)
            # Guard against overwriting input file
            if Path(output_path).resolve() == input_p.resolve():
                output_path = str(output_dir / (input_p.stem + "_converted" + input_p.suffix))
        else:
            output_path = None  # process_file will add _converted suffix

        print(f"Processing: {filepath}")
        try:
            changes = process_file(filepath, output_path, nlp, dry_run=args.dry_run)
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            continue

        if args.dry_run:
            for original, converted in changes:
                print(f"  - {original!r}")
                print(f"  + {converted!r}")
                print()

        if not changes:
            print("  (no changes)")
        else:
            print(f"  {len(changes)} subtitle(s) modified.")


if __name__ == "__main__":
    main()
