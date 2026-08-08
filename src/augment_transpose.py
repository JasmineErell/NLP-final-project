"""
Transpose ABC tunes to all 12 chromatic keys for data augmentation.

This script reads raw ABC files and produces transposed copies of each tune.
Each song gets 11 new versions (one per semitone shift), giving 12x the data.

Usage:
    python -m src.augment_transpose --input ABC_cleaned --output ABC_augmented

Then run the normal preprocessing pipeline on ABC_augmented/.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Music theory constants
# ---------------------------------------------------------------------------

# Chromatic scale using sharps (for sharp keys)
CHROMATIC_SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Chromatic scale using flats (for flat keys)
CHROMATIC_FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Map note names to semitone offset from C
NOTE_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Key signatures: key name -> set of notes that are sharped/flatted
# Major keys
SHARP_KEYS_ORDER = ["F", "C", "G", "D", "A", "E", "B"]  # sharps added in this order
FLAT_KEYS_ORDER = ["B", "E", "A", "D", "G", "C", "F"]   # flats added in this order

KEY_SIGNATURES: dict[str, dict[str, str]] = {}

# Sharp major keys: G(1#), D(2#), A(3#), E(4#), B(5#), F#(6#)
_sharp_notes = []
for i, key in enumerate(["G", "D", "A", "E", "B", "F#"]):
    _sharp_notes = _sharp_notes + [SHARP_KEYS_ORDER[i]]
    KEY_SIGNATURES[key] = {n: "^" for n in _sharp_notes}

# Flat major keys: F(1b), Bb(2b), Eb(3b), Ab(4b), Db(5b), Gb(6b)
_flat_notes = []
for i, key in enumerate(["F", "Bb", "Eb", "Ab", "Db", "Gb"]):
    _flat_notes = _flat_notes + [FLAT_KEYS_ORDER[i]]
    KEY_SIGNATURES[key] = {n: "_" for n in _flat_notes}

# C major has no sharps or flats
KEY_SIGNATURES["C"] = {}

# Minor keys map to their relative major for key signature purposes
MINOR_TO_RELATIVE_MAJOR = {
    "Am": "C", "Em": "G", "Bm": "D", "F#m": "A", "C#m": "E",
    "G#m": "B", "D#m": "F#",
    "Dm": "F", "Gm": "Bb", "Cm": "Eb", "Fm": "Ab", "Bbm": "Db",
    "Ebm": "Gb",
}

# All 12 major key roots in order (by semitone)
ALL_MAJOR_KEYS = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
ALL_MINOR_KEYS = ["Am", "Bbm", "Bm", "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m"]

# Preferred spelling for each semitone (use sharps for sharp keys, flats for flat keys)
# We'll decide based on the target key whether to use sharps or flats
SHARP_KEY_SET = {"C", "G", "D", "A", "E", "B", "F#", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m"}
FLAT_KEY_SET = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm"}


def key_root_to_semitone(key_root: str) -> int:
    """Convert a key root (like 'G', 'Bb', 'F#') to semitone offset from C."""
    if len(key_root) == 1:
        return NOTE_TO_SEMITONE[key_root.upper()]
    elif key_root[1] == "#":
        return (NOTE_TO_SEMITONE[key_root[0].upper()] + 1) % 12
    elif key_root[1] == "b":
        return (NOTE_TO_SEMITONE[key_root[0].upper()] - 1) % 12
    else:
        return NOTE_TO_SEMITONE[key_root[0].upper()]


def semitone_to_note_name(semitone: int, use_sharps: bool) -> str:
    """Convert a semitone (0-11) to a note name."""
    semitone = semitone % 12
    if use_sharps:
        return CHROMATIC_SHARPS[semitone]
    else:
        return CHROMATIC_FLATS[semitone]


def get_key_sig_for_key(key_name: str) -> dict[str, str]:
    """Get the key signature (which notes are altered) for a given key."""
    # Handle minor keys
    if key_name in MINOR_TO_RELATIVE_MAJOR:
        major = MINOR_TO_RELATIVE_MAJOR[key_name]
        return KEY_SIGNATURES.get(major, {})

    return KEY_SIGNATURES.get(key_name, {})


def parse_abc_key(key_field: str) -> tuple[str, str]:
    """
    Parse an ABC K: field into (root, mode).

    Examples:
        "G" -> ("G", "major")
        "Am" -> ("A", "minor")
        "D Mixolydian" -> ("D", "mixolydian")
        "Bb" -> ("Bb", "major")
    """
    key_field = key_field.strip()

    # Handle modes
    mode_map = {
        "m": "minor", "min": "minor", "minor": "minor",
        "maj": "major", "major": "major",
        "mix": "mixolydian", "mixolydian": "mixolydian",
        "dor": "dorian", "dorian": "dorian",
        "phr": "phrygian", "phrygian": "phrygian",
        "lyd": "lydian", "lydian": "lydian",
        "loc": "locrian", "locrian": "locrian",
    }

    # Try to extract root and mode
    # Match patterns like: G, Gm, G minor, G Mixolydian, Bb, Bbm
    match = re.match(
        r"^([A-G][#b]?)\s*(.*)$",
        key_field,
        re.IGNORECASE,
    )

    if not match:
        return ("C", "major")  # fallback

    root = match.group(1)
    # Normalize root capitalization
    root = root[0].upper() + root[1:] if len(root) > 1 else root.upper()

    mode_str = match.group(2).strip().lower()

    if not mode_str:
        return (root, "major")

    for prefix, mode in mode_map.items():
        if mode_str.startswith(prefix):
            return (root, mode)

    return (root, "major")


def transpose_key_name(key_field: str, semitones: int) -> str:
    """Transpose a key field by N semitones."""
    root, mode = parse_abc_key(key_field)
    original_semitone = key_root_to_semitone(root)
    new_semitone = (original_semitone + semitones) % 12

    # Decide whether to use sharp or flat spelling for the new key
    # Use sharps if transposing into a "sharp key" territory
    use_sharps = (new_semitone in [0, 2, 4, 7, 9, 11, 1, 6])  # C,D,E,G,A,B,C#,F#

    new_root = semitone_to_note_name(new_semitone, use_sharps)

    if mode == "minor":
        return f"{new_root}m"
    elif mode == "major":
        return new_root
    else:
        # Capitalize mode for ABC
        return f"{new_root} {mode.capitalize()}"


# ---------------------------------------------------------------------------
# ABC note transposition
# ---------------------------------------------------------------------------

# Regex to match a note in ABC: optional accidentals, note letter, optional octave
ABC_NOTE_RE = re.compile(
    r"(\^{1,2}|_{1,2}|=)?"   # accidental: ^, ^^, _, __, =
    r"([A-Ga-g])"             # note letter (uppercase=low octave, lowercase=high octave)
    r"([',]*)"                # octave modifiers
)

# Regex to match chord symbols in quotes: "G", "Em7", "Bb", "F#dim"
CHORD_RE = re.compile(r'"([^"]*)"')

# Regex for K: field in headers
KEY_FIELD_RE = re.compile(r"^K:\s*(.+)$", re.MULTILINE)


def abc_note_to_semitone(accidental: str | None, letter: str, key_sig: dict[str, str]) -> int:
    """
    Convert an ABC note to an absolute semitone (0-11), considering the key signature.

    In ABC:
    - No accidental: use key signature
    - = : natural (override key sig)
    - ^ : sharp
    - ^^ : double sharp
    - _ : flat
    - __ : double flat
    """
    # Base note (uppercase)
    base_letter = letter.upper()
    base_semitone = NOTE_TO_SEMITONE[base_letter]

    if accidental is None or accidental == "":
        # Use key signature
        if base_letter in key_sig:
            if key_sig[base_letter] == "^":
                return (base_semitone + 1) % 12
            elif key_sig[base_letter] == "_":
                return (base_semitone - 1) % 12
        return base_semitone
    elif accidental == "=":
        # Explicit natural
        return base_semitone
    elif accidental == "^":
        return (base_semitone + 1) % 12
    elif accidental == "^^":
        return (base_semitone + 2) % 12
    elif accidental == "_":
        return (base_semitone - 1) % 12
    elif accidental == "__":
        return (base_semitone - 2) % 12

    return base_semitone


def semitone_to_abc_note(
    target_semitone: int,
    is_lowercase: bool,
    octave_modifiers: str,
    new_key_sig: dict[str, str],
    use_sharps: bool,
) -> str:
    """
    Convert a target semitone back to an ABC note string with appropriate accidentals.

    Returns the note string (accidental + letter + octave modifiers).
    """
    target_semitone = target_semitone % 12

    # Try each natural note to find one that matches (possibly with accidental)
    # Prefer the note that needs no accidental (in the key signature)
    best_result = None

    for base_letter in "CDEFGAB":
        base_sem = NOTE_TO_SEMITONE[base_letter]

        # What semitone does this letter produce under the new key sig?
        if base_letter in new_key_sig:
            if new_key_sig[base_letter] == "^":
                effective_sem = (base_sem + 1) % 12
            else:
                effective_sem = (base_sem - 1) % 12
        else:
            effective_sem = base_sem

        if effective_sem == target_semitone:
            # Perfect match with no accidental needed
            letter = base_letter.lower() if is_lowercase else base_letter
            return f"{letter}{octave_modifiers}"

    # No natural match in key sig — need an accidental
    # Priority: prefer "natural override" (=X) over sharp/flat of adjacent note
    # e.g., =C is better than ^B for the pitch C in a key with C#

    # First try: use a natural sign to override the key signature
    for base_letter in "CDEFGAB":
        base_sem = NOTE_TO_SEMITONE[base_letter]
        if base_sem == target_semitone:
            letter = base_letter.lower() if is_lowercase else base_letter
            if base_letter in new_key_sig:
                return f"={letter}{octave_modifiers}"
            return f"{letter}{octave_modifiers}"

    # Second try: use a sharp or flat
    if use_sharps:
        for base_letter in "CDEFGAB":
            base_sem = NOTE_TO_SEMITONE[base_letter]
            if (base_sem + 1) % 12 == target_semitone:
                letter = base_letter.lower() if is_lowercase else base_letter
                if base_letter in new_key_sig and new_key_sig[base_letter] == "^":
                    # Already sharped in key sig — just write the letter
                    return f"{letter}{octave_modifiers}"
                else:
                    return f"^{letter}{octave_modifiers}"
    else:
        for base_letter in "CDEFGAB":
            base_sem = NOTE_TO_SEMITONE[base_letter]
            if (base_sem - 1) % 12 == target_semitone:
                letter = base_letter.lower() if is_lowercase else base_letter
                if base_letter in new_key_sig and new_key_sig[base_letter] == "_":
                    # Already flatted in key sig — just write the letter
                    return f"{letter}{octave_modifiers}"
                else:
                    return f"_{letter}{octave_modifiers}"

    # Fallback (shouldn't reach here)
    note_name = CHROMATIC_SHARPS[target_semitone]
    if len(note_name) == 1:
        letter = note_name.lower() if is_lowercase else note_name
        return f"{letter}{octave_modifiers}"
    else:
        letter = note_name[0].lower() if is_lowercase else note_name[0]
        return f"^{letter}{octave_modifiers}"


def transpose_note_match(
    match: re.Match,
    semitones: int,
    old_key_sig: dict[str, str],
    new_key_sig: dict[str, str],
    use_sharps: bool,
) -> str:
    """Transpose a single ABC note match by N semitones."""
    accidental = match.group(1) or ""
    letter = match.group(2)
    octave = match.group(3) or ""

    is_lowercase = letter.islower()

    # Get absolute semitone of original note
    original_semitone = abc_note_to_semitone(accidental, letter, old_key_sig)

    # Transpose
    target_semitone = (original_semitone + semitones) % 12

    # Convert back to ABC notation under the new key
    return semitone_to_abc_note(target_semitone, is_lowercase, octave, new_key_sig, use_sharps)


def transpose_chord(chord_text: str, semitones: int, use_sharps: bool) -> str:
    """
    Transpose a chord symbol by N semitones.

    Examples: "G" -> "Bb" (semitones=3, flats)
              "Em7" -> "G#m7" (semitones=4, sharps)
    """
    if not chord_text:
        return chord_text

    # Extract root from chord: first letter + optional # or b
    match = re.match(r"^([A-G][#b]?)(.*)", chord_text)
    if not match:
        return chord_text  # Not a standard chord, return as-is

    root = match.group(1)
    suffix = match.group(2)  # m, 7, dim, m7, etc.

    # Transpose root
    original_semitone = key_root_to_semitone(root)
    new_semitone = (original_semitone + semitones) % 12
    new_root = semitone_to_note_name(new_semitone, use_sharps)

    return f"{new_root}{suffix}"


def transpose_abc_line(
    line: str,
    semitones: int,
    old_key_sig: dict[str, str],
    new_key_sig: dict[str, str],
    use_sharps: bool,
) -> str:
    """
    Transpose all notes and chords in a single ABC music line.

    Handles chord symbols in quotes and note events.
    """
    result = []
    pos = 0

    while pos < len(line):
        # Try chord match
        chord_match = CHORD_RE.match(line, pos)
        if chord_match:
            chord_text = chord_match.group(1)
            transposed_chord = transpose_chord(chord_text, semitones, use_sharps)
            result.append(f'"{transposed_chord}"')
            pos = chord_match.end()
            continue

        # Try note match
        note_match = ABC_NOTE_RE.match(line, pos)
        if note_match:
            # Make sure it's not inside a header field or comment
            transposed = transpose_note_match(
                note_match, semitones, old_key_sig, new_key_sig, use_sharps
            )
            result.append(transposed)
            pos = note_match.end()
            continue

        # Regular character (bar lines, duration numbers, spaces, etc.)
        result.append(line[pos])
        pos += 1

    return "".join(result)


# ---------------------------------------------------------------------------
# Tune-level transposition
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(r"^([A-Za-z]):\s*(.*)$")


def transpose_tune(tune_lines: list[str], semitones: int) -> list[str]:
    """
    Transpose an entire ABC tune by N semitones.

    Returns new list of lines with transposed key, notes, and chords.
    """
    if semitones % 12 == 0:
        return list(tune_lines)  # No transposition needed

    semitones = semitones % 12

    # First pass: find the key
    original_key_field = "C"
    for line in tune_lines:
        stripped = line.strip()
        header_match = HEADER_RE.match(stripped)
        if header_match and header_match.group(1) == "K":
            original_key_field = header_match.group(2).strip()
            break

    root, mode = parse_abc_key(original_key_field)
    original_key_name = f"{root}m" if mode == "minor" else root
    old_key_sig = get_key_sig_for_key(original_key_name)

    # Determine new key
    new_key_field = transpose_key_name(original_key_field, semitones)
    new_root, new_mode = parse_abc_key(new_key_field)
    new_key_name = f"{new_root}m" if new_mode == "minor" else new_root
    new_key_sig = get_key_sig_for_key(new_key_name)

    # Decide sharp vs flat spelling
    use_sharps = new_key_name in SHARP_KEY_SET

    # Second pass: transpose
    output_lines = []
    in_body = False

    for line in tune_lines:
        stripped = line.strip()

        # Handle empty lines
        if not stripped:
            output_lines.append(line)
            continue

        # Handle comments
        if stripped.startswith("%"):
            output_lines.append(line)
            continue

        # Check if it's a header field
        header_match = HEADER_RE.match(stripped)

        if header_match and not in_body:
            field = header_match.group(1)

            if field == "K":
                # Transpose the key field
                # Also handle inline key changes in body
                output_lines.append(f"K:{new_key_field}")
                in_body = True  # K: is always last header field before body
            else:
                output_lines.append(line)
        elif in_body:
            # Check for inline fields (K:, M:, L:, P:, etc.)
            inline_header = HEADER_RE.match(stripped)
            if inline_header:
                field = inline_header.group(1)
                if field == "K":
                    # Inline key change — transpose it
                    inline_key = inline_header.group(2).strip()
                    transposed_inline = transpose_key_name(inline_key, semitones)
                    output_lines.append(f"K:{transposed_inline}")
                    # Update key context for subsequent lines
                    new_root_i, new_mode_i = parse_abc_key(transposed_inline)
                    new_key_name_i = f"{new_root_i}m" if new_mode_i == "minor" else new_root_i
                    new_key_sig = get_key_sig_for_key(new_key_name_i)
                    use_sharps = new_key_name_i in SHARP_KEY_SET

                    old_root_i, old_mode_i = parse_abc_key(inline_key)
                    old_key_name_i = f"{old_root_i}m" if old_mode_i == "minor" else old_root_i
                    old_key_sig = get_key_sig_for_key(old_key_name_i)
                else:
                    # Non-musical fields (P:, M:, L:, etc.) — pass through unchanged
                    output_lines.append(line)
            else:
                # Transpose the music line
                transposed_line = transpose_abc_line(
                    line, semitones, old_key_sig, new_key_sig, use_sharps
                )
                output_lines.append(transposed_line)
        else:
            # Before K: was found — might be a body line if no K: exists
            # Try to transpose anyway
            output_lines.append(line)

    return output_lines


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

TUNE_START_RE = re.compile(r"^X:\s*(.+?)\s*$")


def split_abc_collection(text: str) -> list[list[str]]:
    """Split an ABC file (which may contain multiple tunes) into individual tunes."""
    tunes: list[list[str]] = []
    current: list[str] = []

    for line in text.splitlines():
        if TUNE_START_RE.match(line.strip()):
            if current:
                tunes.append(current)
            current = [line]
        elif current:
            current.append(line)

    if current:
        tunes.append(current)

    return tunes


def augment_file(
    input_path: Path,
    output_dir: Path,
    transpositions: list[int] | None = None,
) -> int:
    """
    Read an ABC file, transpose each tune, write augmented output.

    Returns the total number of tunes written.
    """
    if transpositions is None:
        transpositions = list(range(12))  # 0 through 11 (0 = original)

    text = input_path.read_text(encoding="utf-8", errors="replace")
    tunes = split_abc_collection(text)

    output_tunes: list[str] = []
    tune_count = 0

    for tune_lines in tunes:
        for semitones in transpositions:
            transposed = transpose_tune(tune_lines, semitones)

            # Modify X: and T: to indicate transposition
            new_lines = []
            for line in transposed:
                stripped = line.strip()
                if TUNE_START_RE.match(stripped) and semitones != 0:
                    # Append transposition info to reference number
                    orig_x = stripped.split(":", 1)[1].strip()
                    new_lines.append(f"X: {orig_x}_t{semitones}")
                elif stripped.startswith("T:") and semitones != 0:
                    title = stripped[2:].strip()
                    new_lines.append(f"T:{title} (transposed +{semitones})")
                else:
                    new_lines.append(line)

            output_tunes.append("\n".join(new_lines))
            tune_count += 1

    # Write output
    output_path = output_dir / input_path.name
    output_path.write_text(
        "\n\n".join(output_tunes) + "\n",
        encoding="utf-8",
    )

    return tune_count


def main():
    parser = argparse.ArgumentParser(
        description="Transpose ABC tunes to all 12 keys for data augmentation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ABC_cleaned"),
        help="Directory containing original ABC files (default: ABC_cleaned)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ABC_augmented"),
        help="Output directory for augmented files (default: ABC_augmented)",
    )
    parser.add_argument(
        "--transpositions",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11",
        help="Comma-separated list of semitone transpositions (default: all 12)",
    )

    args = parser.parse_args()

    transpositions = [int(x) for x in args.transpositions.split(",")]

    input_dir = args.input
    output_dir = args.output

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all .abc files
    abc_files = sorted(input_dir.glob("*.abc"))
    if not abc_files:
        print(f"No .abc files found in '{input_dir}'")
        return

    total_tunes = 0
    for abc_file in abc_files:
        count = augment_file(abc_file, output_dir, transpositions)
        print(f"  {abc_file.name}: {count} tunes written")
        total_tunes += count

    print(f"\nDone! {total_tunes} total tunes written to '{output_dir}/'")
    print(f"  (original tunes × {len(transpositions)} transpositions)")


if __name__ == "__main__":
    main()
