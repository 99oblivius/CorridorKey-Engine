"""Frame/clip selection persistence for CorridorKey TUI.

The selection map records which clips and frames the user has
(de)selected.  When everything is selected (the default), no file is
written — the file is deleted to keep the project directory clean.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_SELECTION_VERSION = 1
_FILENAME = ".corridorkey_selection.json"


@dataclass
class SelectionMap:
    """Frame/clip selection state.

    Keys are clip folder names:
    * Absent key → all frames selected.
    * Key with ``None`` → all frames selected (explicit).
    * Key with ``list[int]`` → only those frame indices selected.
    * Key with empty ``[]`` → clip deselected entirely.
    """

    clips: dict[str, list[int] | None] = field(default_factory=dict)

    def is_default(self) -> bool:
        """Return ``True`` when everything is selected (nothing to persist)."""
        return not self.clips

    def save(self, project_root: Path) -> None:
        """Persist selection to JSON, or delete the file if default."""
        dest = project_root / _FILENAME
        if self.is_default():
            dest.unlink(missing_ok=True)
            return
        data = {
            "_version": _SELECTION_VERSION,
            "clips": self.clips,
        }
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(dest)

    @classmethod
    def load(cls, project_root: Path) -> SelectionMap:
        """Load selection from JSON, falling back to default (all selected)."""
        src = project_root / _FILENAME
        if not src.exists():
            return cls()
        try:
            raw = json.loads(src.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse %s, using defaults", src)
            return cls()
        clips = raw.get("clips", {})
        return cls(clips=clips)
