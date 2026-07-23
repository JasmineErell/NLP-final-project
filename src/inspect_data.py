"""Used to inspect the raw dataset before processing everything.
Reads the .abc files
Counts songs in each file
Prints metadata for several songs
Shows the first generated tokens
use with : python3 -m src.pipeline.inspect_data --input ABC_cleaned --samples 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

from abc_utils import iter_abc_files, parse_tune, split_abc_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ABC files and print sample tune metadata."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ABC_cleaned"),
        help="Directory containing .abc files (default: ABC_cleaned).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of parsed tunes to display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total_tunes = 0
    displayed = 0

    print(f"Input directory: {args.input.resolve()}\n")

    for path in iter_abc_files(args.input):
        text = path.read_text(encoding="utf-8", errors="replace")
        tune_texts = split_abc_text(text)
        total_tunes += len(tune_texts)

        print(f"{path.name}: {len(tune_texts)} tunes")

        for tune_text in tune_texts:
            if displayed >= args.samples:
                continue

            try:
                tune = parse_tune(tune_text, path.name)
            except Exception as error:
                print(f"\nCould not parse a sample from {path.name}: {error}")
                continue

            print("\n" + "=" * 70)
            print(f"Song ID: {tune.song_id}")
            print(f"Title: {tune.titles[0]}")
            print(f"Rhythm: {tune.rhythm}")
            print(f"Meter: {tune.meter}")
            print(f"Default length: {tune.default_note_length}")
            print(f"Key: {tune.key}")
            print(f"Chord annotations: {len(tune.quoted_symbols)}")
            print("First tokens:")
            for token in tune.sequence_tokens[:8]:
                print(f"  {token}")

            displayed += 1

    print("\n" + "=" * 70)
    print(f"Total tunes found: {total_tunes}")
    print(f"Samples displayed: {displayed}")


if __name__ == "__main__":
    main()
