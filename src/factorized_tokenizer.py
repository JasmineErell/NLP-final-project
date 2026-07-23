from __future__ import annotations

import re


MUSIC_EVENT_RE = re.compile(
    r"""
    \[\d+                       | # first/second ending
    \|:|:\||\|\]|\[\||\|\|?    | # bars and repeats
    \(\d+                       | # tuplets
    [\^_=]*                     # accidentals
    [A-Ga-gzZxX]                # note or rest
    [',]*                       # octave symbols
    (?:
        \d+/\d+                 |
        \d+                     |
        /+
    )?                            # optional duration
    """,
    re.VERBOSE,
)


def tokenize_melody_events(melody: str) -> list[str]:
    """Split an ABC melody into reusable event tokens."""
    tokens: list[str] = []
    previous_end = 0

    for match in MUSIC_EVENT_RE.finditer(melody):
        text_between = melody[previous_end:match.start()]

        if " " in text_between:
            tokens.append("[GROUP_BOUNDARY]")

        event = match.group(0)

        bar_mapping = {
            "|": "[BAR]",
            "||": "[DOUBLE_BAR]",
            "|:": "[REPEAT_START]",
            ":|": "[REPEAT_END]",
            "|]": "[FINAL_BAR]",
            "[|": "[DOUBLE_BAR_START]",
        }

        if event in bar_mapping:
            tokens.append(bar_mapping[event])
        else:
            tokens.append(f"[EVENT_{event}]")

        previous_end = match.end()

    return tokens


def create_factorized_sequence(song: dict) -> list[str]:
    """Create a factorized sequence for one processed song."""
    sequence = list(song["metadata_tokens"])

    for segment in song["segments"]:
        sequence.append("[SEGMENT_START]")
        sequence.append(f"[CHORD_{segment['chord']}]")

        sequence.extend(
            tokenize_melody_events(
                segment["melody_normalized"]
            )
        )

        sequence.append("[SEGMENT_END]")

    return sequence