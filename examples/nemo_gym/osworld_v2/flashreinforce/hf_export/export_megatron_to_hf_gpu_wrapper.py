"""Run examples/converters/convert_megatron_to_hf_gpu.py with read-only HF source artifacts.

The reference HF export is chmod 444; shutil.copy2 preserves that mode, so the Omni bridge's
postprocess (which appends to modeling.py) fails with PermissionError on rank 0 while the other
ranks wait in the save barrier forever.
"""
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, "/opt/nemo-rl/examples/converters")

from megatron.bridge.models.nemotron_omni.nemotron_omni_bridge import NemotronOmniBridge

import convert_megatron_to_hf_gpu

_orig_postprocess = NemotronOmniBridge.postprocess_hf_export_artifacts


def _postprocess_writable(self, path: Path) -> None:
    for p in Path(path).iterdir():
        if p.is_file():
            os.chmod(p, p.stat().st_mode | stat.S_IWUSR)
    _orig_postprocess(self, path)


NemotronOmniBridge.postprocess_hf_export_artifacts = _postprocess_writable

if __name__ == "__main__":
    convert_megatron_to_hf_gpu.main()
