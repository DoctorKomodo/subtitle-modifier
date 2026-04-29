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
        "-v", "--verbose",
        action="store_true",
        help="Show debug logging (e.g. raw LLM responses on parse failures).",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        metavar="MODEL",
        help="Benchmark one or more spaCy models and print a speed comparison table.",
    )

    # LLM mode arguments
    llm_group = parser.add_argument_group("LLM mode", "Use an LLM instead of spaCy NER for recasing.")
    llm_group.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM mode (uses OpenAI-compatible API instead of spaCy).",
    )
    llm_group.add_argument(
        "--llm-model",
        help="LLM model name (required when --llm is used).",
    )
    llm_group.add_argument(
        "--llm-url",
        default="http://localhost:11434/v1",
        help="Base URL for OpenAI-compatible API (default: http://localhost:11434/v1 for Ollama).",
    )
    llm_group.add_argument(
        "--llm-api-key",
        default=None,
        help="API key (default: OPENAI_API_KEY env var, falls back to 'ollama').",
    )
    llm_group.add_argument(
        "--llm-batch-size",
        type=int,
        default=50,
        help="Number of subtitle events per LLM API call (default: 50).",
    )

    # Claude mode arguments
    claude_group = parser.add_argument_group(
        "Claude mode",
        "Use the native Anthropic SDK for recasing.",
    )
    claude_group.add_argument(
        "--claude",
        action="store_true",
        help="Enable Claude mode (uses the native Anthropic SDK).",
    )
    claude_group.add_argument(
        "--claude-model",
        default="claude-haiku-4-5",
        help="Claude model ID (default: claude-haiku-4-5).",
    )
    claude_group.add_argument(
        "--claude-api-key",
        default=None,
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var).",
    )
    claude_group.add_argument(
        "--claude-batch-size",
        type=int,
        default=50,
        help="Number of subtitle events per Claude API call (default: 50).",
    )

    args = parser.parse_args(argv)

    if args.llm and args.claude:
        print("Error: --llm and --claude are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")

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

    from .subtitle_io import process_file

    # Set up conversion backend
    convert_fn = None
    nlp = None

    if args.claude:
        try:
            import anthropic
        except ImportError:
            print(
                "Error: anthropic package not installed. "
                "Install it with: pip install 'subtitle-modifier[claude]'",
                file=sys.stderr,
            )
            sys.exit(1)

        import os

        from .claude import convert_texts_claude

        api_key = args.claude_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "Error: no Anthropic API key. Pass --claude-api-key or set "
                "ANTHROPIC_API_KEY in the environment.",
                file=sys.stderr,
            )
            sys.exit(1)

        client = anthropic.Anthropic(api_key=api_key)

        def convert_fn(
            texts,
            _client=client,
            _model=args.claude_model,
            _bs=args.claude_batch_size,
        ):
            return convert_texts_claude(texts, _client, _model, batch_size=_bs)

    elif args.llm:
        if not args.llm_model:
            print("Error: --llm-model is required when using --llm mode.", file=sys.stderr)
            sys.exit(1)

        try:
            import openai
        except ImportError:
            print(
                "Error: openai package not installed. "
                "Install it with: pip install 'subtitle-modifier[llm]'",
                file=sys.stderr,
            )
            sys.exit(1)

        import os

        from .llm import convert_texts_llm

        api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY", "ollama")
        client = openai.OpenAI(base_url=args.llm_url, api_key=api_key)

        def convert_fn(texts, _client=client, _model=args.llm_model, _bs=args.llm_batch_size):
            return convert_texts_llm(texts, _client, _model, batch_size=_bs)
    else:
        # Lazy-load spaCy (heavy import)
        import spacy

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
            changes = process_file(
                filepath, output_path, nlp, convert_fn=convert_fn, dry_run=args.dry_run,
            )
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
