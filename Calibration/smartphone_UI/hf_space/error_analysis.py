"""
Error analysis for MeTRAbs pose estimation output.

This file is a STUB. Teammate: fill in the `analyze()` function body with
the real error metrics. Keep the function signature the same so app.py
keeps working.

Input CSV columns (from MeTRAbs):
    frame       int    frame index (1-based)
    person_id   int    0, 1, ... (multi-person)
    joint_id    int    0..23 for smpl_24 skeleton
    x, y, z     float  3D position in mm (if intrinsics were used)
                       otherwise in MeTRAbs canonical scale

Run standalone to test on an existing CSV:
    python error_analysis.py path/to/poses.csv
"""
from __future__ import annotations

from pathlib import Path


def analyze(csv_path: str | Path, used_intrinsics: bool = False) -> dict:
    """Compute error metrics from a MeTRAbs CSV.

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file produced by MeTRAbs.
    used_intrinsics : bool
        True if the video was processed with ARKit camera intrinsics
        (so x/y/z are in real-world mm). False if uncalibrated.

    Returns
    -------
    dict
        Analysis results. Must be JSON-serializable (no numpy, no pandas
        objects). Example shape:
            {
                "n_frames": 30,
                "n_joints_tracked": 24,
                "mean_joint_error_mm": 12.4,
                ...
            }
        On failure, return {"error": "<message>"}.
    """
    # TODO: teammate — replace this stub with real analysis.
    return {
        "status": "stub - error_analysis.py not yet implemented",
        "used_intrinsics": used_intrinsics,
        "csv_path": str(csv_path),
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("usage: python error_analysis.py <csv_path> [used_intrinsics=0|1]")
        sys.exit(1)

    csv_p = sys.argv[1]
    flag = bool(int(sys.argv[2])) if len(sys.argv) >= 3 else False
    print(json.dumps(analyze(csv_p, flag), indent=2))
