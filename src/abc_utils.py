"""

"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Iterator


HEADER_RE = re.compile(r"^([A-Za-z]):\s*(.*)$")
TUNE_START_RE = re.compile(r"^X:\s*(.+?)\s*$")
QUOTED_SYMBOL_RE = re.compile(r'"([^"\n]+)"')


@dataclass
class ParsedTune:
    """A parsed tune before it is serialized to JSON."""

    song_id: str
    source_file: str
    reference_number: str
    titles: list[str]
    rhythm: str | None
    meter: str | None
    default_note_length: str | None
    key: str | None
    raw_body: str
    metadata_tokens: list[str]
    segment_tokens: list[str]
    sequence_tokens: list[str]
    has_chords: bool
    usable_for_chord_model: bool
    quoted_symbols: list[str]


def strip_inline_comment(line: str) -> str:
    """Remove an ABC '%' comment unless the percent sign is inside quotes."""
    inside_quotes = False
    output: list[str] = []

    for character in line:
        if character == '"':
            inside_quotes = not inside_quotes
            output.append(character)
        elif character == "%" and not inside_quotes:
            break
        else:
            output.append(character)

    return "".join(output).rstrip()


def split_abc_text(text: str) -> list[str]:
    """Split a collection-style ABC file into individual tunes using X: lines."""
    tunes: list[list[str]] = []
    current: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")

        if TUNE_START_RE.match(line.strip()):
            if current:
                tunes.append(current)
            current = [line]
        elif current:
            current.append(line)

    if current:
        tunes.append(current)

    return ["\n".join(lines).strip() for lines in tunes if lines]


def safe_component(value: str) -> str:
    """Convert metadata text into a stable token component."""
    value = value.strip().upper()
    value = value.replace("/", "_")
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Z0-9_#+.\-]", "", value)
    return value or "UNKNOWN"


def normalize_chord(chord: str) -> str:
    """Normalize whitespace without changing the musical identity of a chord."""
    chord = chord.strip()
    chord = re.sub(r"\s+", "", chord)
    return chord or "UNKNOWN"


def normalize_melody(melody: str) -> str:
    """Normalize spacing and common structural symbols inside a melody segment.

    Notes, accidentals, octaves, durations, tuplets, slurs and ties are retained.
    Whitespace is removed so a complete chord-melody segment is one token.
    """
    melody = melody.replace("\\", "")
    melody = re.sub(r"\s+", "", melody)

    # Replace longer symbols before shorter ones.
    replacements = [
        ("|:", "<REPEAT_START>"),
        (":|", "<REPEAT_END>"),
        ("|]", "<FINAL_BAR>"),
        ("[|", "<DOUBLE_BAR_START>"),
        ("||", "<DOUBLE_BAR>"),
        ("|", "<BAR>"),
    ]
    for source, target in replacements:
        melody = melody.replace(source, target)

    return melody.strip()


def contains_musical_content(text: str) -> bool:
    """Return True when a text fragment contains notes/rests or ABC structure."""
    compact = re.sub(r"\s+", "", text)
    compact = re.sub(r"\[PART_[^\]]+\]", "", compact)
    return bool(re.search(r"[A-Ga-gzZxX\^_=|:()\[\]<>0-9]", compact))


def convert_body_field(line: str) -> str | None:
    """Convert selected body fields into explicit sequence markers."""
    match = HEADER_RE.match(line.strip())
    if not match:
        return None

    field, value = match.groups()
    field = field.upper()
    component = safe_component(value)

    if field == "P":
        return f"[PART_{component}]"
    if field == "K":
        return f"[KEY_CHANGE_{component}]"
    if field == "M":
        return f"[METER_CHANGE_{component}]"
    if field == "L":
        return f"[LENGTH_CHANGE_{component}]"
    if field == "V":
        return f"[VOICE_{component}]"

    # Lyrics, comments and descriptive fields are not musical events.
    return ""


def prepare_music_body(lines: Iterable[str]) -> str:
    """Clean body lines while preserving section/key/meter changes."""
    prepared: list[str] = []

    for raw_line in lines:
        line = strip_inline_comment(raw_line).strip()
        if not line:
            continue

        field_marker = convert_body_field(line)
        if field_marker is not None:
            if field_marker:
                prepared.append(field_marker)
            continue

        # Ignore common ABC directives.
        if line.startswith("%%"):
            continue

        prepared.append(line)

    return "\n".join(prepared)


def extract_chord_melody_segments(body: str) -> tuple[list[str], list[str]]:
    """Create one token per quoted chord and its following melody fragment.

    Melody before the first chord is retained as CHORD_NONE when the tune later
    contains chord annotations. Songs with no quoted chord symbols are marked
    unusable by the caller instead of becoming one enormous unique token.
    """
    matches = list(QUOTED_SYMBOL_RE.finditer(body))
    quoted_symbols = [match.group(1).strip() for match in matches]

    if not matches:
        return [], []

    segments: list[str] = []

    prefix = body[: matches[0].start()]
    prefix = normalize_melody(prefix)
    if prefix and contains_musical_content(prefix):
        segments.append(f"CHORD_NONE__MELODY_{prefix}")

    for index, match in enumerate(matches):
        chord = normalize_chord(match.group(1))
        melody_start = match.end()
        melody_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(body)
        )

        melody = normalize_melody(body[melody_start:melody_end])
        if not melody:
            melody = "<EMPTY>"

        segments.append(f"CHORD_{chord}__MELODY_{melody}")

    return segments, quoted_symbols


def parse_tune(tune_text: str, source_file: str) -> ParsedTune:
    """Parse a single tune and create metadata and chord-melody tokens."""
    raw_lines = tune_text.splitlines()
    if not raw_lines:
        raise ValueError("Tune is empty.")

    headers: dict[str, list[str]] = {}
    body_lines: list[str] = []
    body_started = False

    for raw_line in raw_lines:
        line = strip_inline_comment(raw_line).strip()
        if not line:
            if body_started:
                body_lines.append("")
            continue

        match = HEADER_RE.match(line)

        if not body_started and match:
            field, value = match.groups()
            field = field.upper()
            headers.setdefault(field, []).append(value.strip())

            # In ABC, the first K: line conventionally ends the header.
            if field == "K":
                body_started = True
            continue

        if body_started:
            body_lines.append(raw_line)

    reference_number = (headers.get("X") or ["UNKNOWN"])[0]
    titles = headers.get("T") or ["Untitled"]
    rhythm = (headers.get("R") or [None])[0]
    meter = (headers.get("M") or [None])[0]
    default_note_length = (headers.get("L") or [None])[0]
    key = (headers.get("K") or [None])[0]

    file_stem = Path(source_file).stem
    song_id = f"{file_stem}_{safe_component(reference_number)}"

    body = prepare_music_body(body_lines)
    segment_tokens, quoted_symbols = extract_chord_melody_segments(body)

    metadata_tokens: list[str] = []
    if key:
        metadata_tokens.append(f"[KEY_{safe_component(key)}]")
    if rhythm:
        metadata_tokens.append(f"[RHYTHM_{safe_component(rhythm)}]")
    if meter:
        metadata_tokens.append(f"[METER_{safe_component(meter)}]")
    if default_note_length:
        metadata_tokens.append(
            f"[DEFAULT_LENGTH_{safe_component(default_note_length)}]"
        )

    has_chords = bool(quoted_symbols)
    usable = has_chords and bool(segment_tokens)
    sequence_tokens = metadata_tokens + segment_tokens if usable else []

    return ParsedTune(
        song_id=song_id,
        source_file=source_file,
        reference_number=reference_number,
        titles=titles,
        rhythm=rhythm,
        meter=meter,
        default_note_length=default_note_length,
        key=key,
        raw_body=body,
        metadata_tokens=metadata_tokens,
        segment_tokens=segment_tokens,
        sequence_tokens=sequence_tokens,
        has_chords=has_chords,
        usable_for_chord_model=usable,
        quoted_symbols=quoted_symbols,
    )


def iter_abc_files(input_directory: Path) -> Iterator[Path]:
    """Yield ABC files in a deterministic order."""
    if not input_directory.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_directory}"
        )

    files = sorted(input_directory.glob("*.abc"))
    if not files:
        raise FileNotFoundError(
            f"No .abc files were found in: {input_directory}"
        )

    yield from files


def tune_to_dict(tune: ParsedTune) -> dict:
    """Convert ParsedTune to a JSON-serializable dictionary."""
    return {
        "song_id": tune.song_id,
        "source_file": tune.source_file,
        "reference_number": tune.reference_number,
        "titles": tune.titles,
        "title": tune.titles[0],
        "rhythm": tune.rhythm,
        "meter": tune.meter,
        "default_note_length": tune.default_note_length,
        "key": tune.key,
        "has_chords": tune.has_chords,
        "usable_for_chord_model": tune.usable_for_chord_model,
        "metadata_tokens": tune.metadata_tokens,
        "segment_tokens": tune.segment_tokens,
        "sequence_tokens": tune.sequence_tokens,
        "number_of_segments": len(tune.segment_tokens),
        "quoted_symbols": tune.quoted_symbols,
        "raw_body": tune.raw_body,
    }


def token_statistics(songs: Iterable[dict]) -> tuple[Counter, dict]:
    """Calculate segment-token frequency statistics."""
    songs = list(songs)
    usable = [song for song in songs if song["usable_for_chord_model"]]
    frequencies = Counter(
        token
        for song in usable
        for token in song["segment_tokens"]
    )

    total_tokens = sum(frequencies.values())
    unique_tokens = len(frequencies)
    singleton_count = sum(1 for count in frequencies.values() if count == 1)
    rare_under_five = sum(1 for count in frequencies.values() if count < 5)

    segment_lengths = [song["number_of_segments"] for song in usable]
    average_segments = (
        sum(segment_lengths) / len(segment_lengths) if segment_lengths else 0.0
    )

    stats = {
        "total_songs": len(songs),
        "songs_with_chords": sum(song["has_chords"] for song in songs),
        "songs_without_chords": sum(not song["has_chords"] for song in songs),
        "usable_songs": len(usable),
        "total_segment_tokens": total_tokens,
        "unique_segment_tokens": unique_tokens,
        "tokens_appearing_once": singleton_count,
        "tokens_appearing_fewer_than_five_times": rare_under_five,
        "singleton_rate": (
            singleton_count / unique_tokens if unique_tokens else 0.0
        ),
        "average_segments_per_usable_song": average_segments,
        "minimum_segments": min(segment_lengths) if segment_lengths else 0,
        "maximum_segments": max(segment_lengths) if segment_lengths else 0,
    }

    return frequencies, stats
