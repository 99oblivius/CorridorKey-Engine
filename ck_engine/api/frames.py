"""Frame range parsing and formatting for the CorridorKey engine protocol.

Frame ranges use 1-based inclusive syntax in the protocol (matching
user-facing frame numbers) and are converted to 0-based indices internally.

Syntax examples:
    None        -> all frames
    "1-100"     -> frames 1 through 100 (inclusive)
    "1,5,10-20" -> frames 1, 5, and 10 through 20
    "50-"       -> frame 50 to end
"""

from __future__ import annotations


def parse_frame_range(spec: str | None, total: int) -> list[int]:
    """Parse a frame range specification into a sorted list of 0-based indices.

    Frame numbers in the spec are 1-based and inclusive; the returned indices
    are 0-based.  All results are clamped to [0, total) and deduplicated.

    Args:
        spec: Frame range string, or None / empty string for all frames.
            Supported forms:
                ``"N"``   — single frame (1-based)
                ``"N-M"`` — inclusive range (1-based, M >= N)
                ``"N-"``  — from N to the last frame
            Multiple parts are separated by commas, e.g. ``"1,5,10-20,50-"``.
        total: Total number of frames in the clip.  Must be > 0.

    Returns:
        Sorted list of unique 0-based frame indices within [0, total).

    Raises:
        ValueError: If *total* <= 0.
        ValueError: If any frame number is not a positive integer.
        ValueError: If the end of a range is less than the start.

    Examples:
        >>> parse_frame_range(None, 5)
        [0, 1, 2, 3, 4]
        >>> parse_frame_range("", 5)
        [0, 1, 2, 3, 4]
        >>> parse_frame_range("1-3", 5)
        [0, 1, 2]
        >>> parse_frame_range("1,5,10-20", 25)
        [0, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        >>> parse_frame_range("3-", 5)
        [2, 3, 4]
        >>> parse_frame_range("2-2", 5)
        [1]
    """
    if total <= 0:
        raise ValueError(f"total must be > 0, got {total!r}")

    if not spec:
        return list(range(total))

    indices: set[int] = set()

    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            # Could be "N-M", "N-", or even a negative number like "-5" which
            # we treat as invalid (negative frame numbers are not meaningful).
            dash_pos = part.index("-")

            if dash_pos == 0:
                # Leading dash: "-M" or just "-" — not valid syntax.
                raise ValueError(
                    f"Invalid frame range part {part!r}: "
                    "frame numbers must be positive integers"
                )

            start_str = part[:dash_pos]
            end_str = part[dash_pos + 1:]

            start_1based = _parse_positive_int(start_str, part)
            start_0based = start_1based - 1

            if end_str == "":
                # Open range "N-"
                end_0based = total - 1
            else:
                end_1based = _parse_positive_int(end_str, part)
                if end_1based < start_1based:
                    raise ValueError(
                        f"Invalid frame range {part!r}: "
                        f"end ({end_1based}) is less than start ({start_1based})"
                    )
                end_0based = end_1based - 1

            for idx in range(start_0based, end_0based + 1):
                clamped = max(0, min(idx, total - 1))
                indices.add(clamped)

        else:
            # Single frame number
            n_1based = _parse_positive_int(part, part)
            n_0based = n_1based - 1
            clamped = max(0, min(n_0based, total - 1))
            indices.add(clamped)

    return sorted(indices)


def format_frame_range(indices: list[int]) -> str:
    """Convert a list of 0-based frame indices to a 1-based range string.

    Consecutive indices are collapsed into ``"start-end"`` ranges; isolated
    indices are emitted as plain numbers.  The output uses the same syntax
    accepted by :func:`parse_frame_range`.

    Args:
        indices: 0-based frame indices.  Need not be sorted or deduplicated;
            the function sorts and deduplicates internally.

    Returns:
        Compact range string, or ``""`` for an empty list.

    Examples:
        >>> format_frame_range([])
        ''
        >>> format_frame_range([0])
        '1'
        >>> format_frame_range([0, 1, 2, 4, 9, 10, 11])
        '1-3,5,10-12'
        >>> format_frame_range([0, 4])
        '1,5'
        >>> format_frame_range([3, 0, 1, 3])
        '1-2,4'
    """
    if not indices:
        return ""

    unique_sorted = sorted(set(indices))

    # Group consecutive indices into runs.
    runs: list[tuple[int, int]] = []
    run_start = unique_sorted[0]
    run_end = unique_sorted[0]

    for idx in unique_sorted[1:]:
        if idx == run_end + 1:
            run_end = idx
        else:
            runs.append((run_start, run_end))
            run_start = idx
            run_end = idx
    runs.append((run_start, run_end))

    parts: list[str] = []
    for start, end in runs:
        start_1based = start + 1
        end_1based = end + 1
        if start == end:
            parts.append(str(start_1based))
        else:
            parts.append(f"{start_1based}-{end_1based}")

    return ",".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_positive_int(value: str, context: str) -> int:
    """Parse *value* as a positive (>= 1) integer.

    Args:
        value: String to parse.
        context: The original range part, used in error messages.

    Returns:
        The parsed integer.

    Raises:
        ValueError: If *value* is not a valid integer or is < 1.
    """
    try:
        n = int(value)
    except ValueError:
        raise ValueError(
            f"Invalid frame range part {context!r}: "
            f"{value!r} is not a valid integer"
        )
    if n < 1:
        raise ValueError(
            f"Invalid frame range part {context!r}: "
            f"frame numbers must be >= 1, got {n}"
        )
    return n
