"""Convert Nottingham ABC collection files into a tokenized JSONL corpus."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import traceback

from abc_utils import (
    iter_abc_files,
    parse_tune,
    split_abc_text,
    token_statistics,
    tune_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse ABC tunes and create chord-melody segment tokens."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ABC_cleaned"),
        help="Directory containing .abc files (default: ABC_cleaned).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory (default: data/processed).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of tunes for a small pilot run.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Number of original/tokenized examples to save.",
    )
    return parser.parse_args()


def save_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_examples(songs: list[dict], path: Path, count: int) -> None:
    usable = [song for song in songs if song["usable_for_chord_model"]]

    with path.open("w", encoding="utf-8") as output_file:
        for song in usable[:count]:
            output_file.write("=" * 80 + "\n")
            output_file.write(f"SONG ID: {song['song_id']}\n")
            output_file.write(f"TITLE: {song['title']}\n")
            output_file.write(
                f"KEY: {song['key']} | RHYTHM: {song['rhythm']} "
                f"| METER: {song['meter']}\n\n"
            )
            output_file.write("ORIGINAL BODY:\n")
            output_file.write(song["raw_body"] + "\n\n")
            output_file.write("TOKENS:\n")
            for token in song["sequence_tokens"]:
                output_file.write(token + "\n")
            output_file.write("\n")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    songs: list[dict] = []
    errors: list[dict] = []
    processed_count = 0

    for abc_path in iter_abc_files(args.input):
        text = abc_path.read_text(encoding="utf-8", errors="replace")
        tune_texts = split_abc_text(text)

        for tune_index, tune_text in enumerate(tune_texts, start=1):
            if args.limit is not None and processed_count >= args.limit:
                break

            try:
                parsed = parse_tune(tune_text, abc_path.name)
                songs.append(tune_to_dict(parsed))
            except Exception as error:
                errors.append(
                    {
                        "source_file": abc_path.name,
                        "tune_index_in_file": tune_index,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )

            processed_count += 1

        if args.limit is not None and processed_count >= args.limit:
            break

    frequencies, stats = token_statistics(songs)
    stats["parse_errors"] = len(errors)
    stats["input_directory"] = str(args.input)
    stats["output_directory"] = str(args.output)

    save_jsonl(songs, args.output / "songs.jsonl")
    save_jsonl(errors, args.output / "errors.jsonl")
    save_examples(
        songs,
        args.output / "tokenization_examples.txt",
        args.examples,
    )

    with (args.output / "stats.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, ensure_ascii=False)

    with (args.output / "token_frequencies.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["token", "count"])
        for token, count in frequencies.most_common():
            writer.writerow([token, count])

    print("\nPreprocessing complete")
    print("-" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nCreated:")
    print(f"  {args.output / 'songs.jsonl'}")
    print(f"  {args.output / 'stats.json'}")
    print(f"  {args.output / 'token_frequencies.csv'}")
    print(f"  {args.output / 'tokenization_examples.txt'}")
    print(f"  {args.output / 'errors.jsonl'}")


if __name__ == "__main__":
    main()
