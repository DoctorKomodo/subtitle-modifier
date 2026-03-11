"""Benchmark subtitle processing across different spaCy models."""

import sys
import time
from dataclasses import dataclass

import pysubs2

from .converter import convert_text


@dataclass
class BenchmarkResult:
    model: str
    load_time: float
    process_time: float
    subtitle_count: int

    @property
    def total_time(self) -> float:
        return self.load_time + self.process_time

    @property
    def subs_per_sec(self) -> float:
        return self.subtitle_count / self.process_time if self.process_time > 0 else 0.0


def run_benchmark(files: list[str], models: list[str]) -> list[BenchmarkResult]:
    """Benchmark each spaCy model against the given subtitle files.

    Returns a list of BenchmarkResult, one per model.
    """
    import spacy

    # Pre-load all subtitle events once
    all_events: list[str] = []
    for path in files:
        subs = pysubs2.load(path)
        for event in subs.events:
            if not event.is_drawing:
                all_events.append(event.text)

    results = []
    for model_name in models:
        # Time model loading
        t0 = time.perf_counter()
        try:
            nlp = spacy.load(model_name)
        except OSError:
            print(f"  Skipping '{model_name}': model not installed.", file=sys.stderr)
            continue
        load_time = time.perf_counter() - t0

        # Time processing
        t0 = time.perf_counter()
        for text in all_events:
            convert_text(text, nlp)
        process_time = time.perf_counter() - t0

        results.append(BenchmarkResult(
            model=model_name,
            load_time=load_time,
            process_time=process_time,
            subtitle_count=len(all_events),
        ))

    return results


def print_results(results: list[BenchmarkResult], file_count: int) -> None:
    """Print a formatted benchmark comparison table."""
    if not results:
        print("No benchmark results to display.")
        return

    sub_count = results[0].subtitle_count
    print(f"\nBenchmark: {sub_count} subtitles from {file_count} file(s)\n")

    # Header
    header = f"{'Model':<25} {'Load (s)':>10} {'Process (s)':>13} {'Total (s)':>11} {'Subs/sec':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.model:<25} {r.load_time:>10.2f} {r.process_time:>13.2f} "
            f"{r.total_time:>11.2f} {r.subs_per_sec:>10.1f}"
        )
