from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


class NotebookOutputCleaner:
    """Remove the Kaggle source checkout while leaving notebook outputs intact."""

    SOURCE_CHECKOUT_NAME = "mdv_asr_source"

    def __init__(self, working_dir: str | Path = "/kaggle/working"):
        self.working_dir = Path(working_dir).expanduser().resolve()

    def remove_source_checkout(
        self,
        source_dir: str | Path,
        *,
        preserve_paths: Iterable[str | Path] = (),
    ) -> bool:
        """Remove only ``mdv_asr_source`` directly under the configured working dir.

        Returns False when the checkout is already absent. Raises ValueError if
        the target is outside the allowed location or contains a protected output.
        """
        requested = Path(source_dir).expanduser()
        if requested.is_symlink():
            raise ValueError("Refusing to remove a symlink as the source checkout")

        target = requested.resolve()
        if (target.name != self.SOURCE_CHECKOUT_NAME
                or target.parent != self.working_dir):
            raise ValueError(
                f"Source checkout must be a direct child named "
                f"{self.SOURCE_CHECKOUT_NAME!r} of {self.working_dir}"
            )

        for output in preserve_paths:
            protected = Path(output).expanduser().resolve()
            if protected == target or target in protected.parents:
                raise ValueError(
                    f"Refusing to remove source checkout because it contains "
                    f"a protected output: {protected}"
                )

        if not target.exists():
            return False
        if not target.is_dir():
            raise ValueError(f"Source checkout is not a directory: {target}")

        shutil.rmtree(target)
        return True
