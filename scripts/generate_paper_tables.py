from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers (from original script)
# ---------------------------------------------------------------------------

def _parse_optional_int(raw: Any) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() == "none":
        return None
    return int(float(text))


def _safe_pct(num: int, den: int) -> float | None:
    if int(den) <= 0:
        return None
    return float(num) / float(den)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{str(k): ("" if v is None else str(v)) for k, v in row.items()} for row in csv.DictReader(handle)]


def _parse_timeline(raw: str) -> list[int | None]:
    text = str(raw or "").strip()
    if not text:
        return []
    value = ast.literal_eval(text)
    if not isinstance(value, list):
        return []
    out: list[int | None] = []
    for item in value:
        if item is None:
            out.append(None)
        else:
            out.append(int(item))
    return out


def _split_visibility_segments(frames_timeline: list[int]) -> list[tuple[int, int]]:
    if not frames_timeline:
        return []
    segments: list[tuple[int, int]] = []
    start_idx = 0
    prev_frame = int(frames_timeline[0])
    for idx in range(1, len(frames_timeline)):
        cur_frame = int(frames_timeline[idx])
        if cur_frame > prev_frame + 1:
            segments.append((start_idx, idx - 1))
            start_idx = idx
        prev_frame = cur_frame
    segments.append((start_idx, len(frames_timeline) - 1))
    return segments


# ---------------------------------------------------------------------------
# Parse report.txt
# ---------------------------------------------------------------------------

def _parse_report(path: Path) -> dict[str, str]:
    """Parse key-value pairs from report.txt [BATCH][Summary] section."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    # Extract only the [BATCH][Summary] section
    match = re.search(r"\[BATCH\]\[Summary\]\s*\n(.*?)(?:\n\[BATCH\]|\Z)", text, re.DOTALL)
    if not match:
        return {}
    kv: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        kv[key.strip()] = val.strip()
    return kv


def _report_float(report: dict[str, str], key: str) -> float | None:
    """Extract a float from a report value like '66.67%' or '0.3885'."""
    raw = report.get(key, "").strip()
    if not raw:
        return None
    raw = raw.rstrip("%").rstrip("x").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def _report_int(report: dict[str, str], key: str) -> int | None:
    raw = report.get(key, "").strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Parse failed_scenes.txt
# ---------------------------------------------------------------------------

def _parse_failed_scenes(path: Path) -> list[str]:
    """Parse comma-separated scene names from failed_scenes.txt."""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [s.strip() for s in text.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Count n_cases from per_case.csv (row count)
# ---------------------------------------------------------------------------

def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # skip header
        return sum(1 for _ in reader)


# ---------------------------------------------------------------------------
# Reappearance analysis (extracted from original script)
# ---------------------------------------------------------------------------

def _build_reappearance_events_for_object(
    row: dict[str, str],
    *,
    scene_name: str,
) -> list[dict[str, Any]]:
    gt_id = _parse_optional_int(row.get("gt_instance_id"))
    if gt_id is None:
        return []

    reference_pred_id = _parse_optional_int(row.get("reference_pred_id"))
    pred_ids_timeline = _parse_timeline(row.get("pred_ids_timeline", ""))
    frames_timeline = _parse_timeline(row.get("frames_timeline", ""))
    if len(pred_ids_timeline) != len(frames_timeline):
        return []

    visibility_segments = _split_visibility_segments(
        [int(frame) for frame in frames_timeline if frame is not None]
    )
    events: list[dict[str, Any]] = []
    for reappearance_index, (start_idx, end_idx) in enumerate(visibility_segments[1:], start=1):
        segment_frames = [int(frame) for frame in frames_timeline[start_idx : end_idx + 1] if frame is not None]
        segment_preds = pred_ids_timeline[start_idx : end_idx + 1]
        if not segment_frames:
            continue

        first_pred_id = segment_preds[0]
        first_ok_frame = None
        if reference_pred_id is not None:
            for frame_id, pred_id in zip(segment_frames, segment_preds):
                if pred_id is not None and int(pred_id) == int(reference_pred_id):
                    first_ok_frame = int(frame_id)
                    break

        perfect_recovery = bool(
            reference_pred_id is not None
            and first_pred_id is not None
            and int(first_pred_id) == int(reference_pred_id)
        )
        permissive_recovery = first_ok_frame is not None

        events.append(
            {
                "scene_name": scene_name,
                "gt_instance_id": int(gt_id),
                "perfect_recovery": bool(perfect_recovery),
                "permissive_recovery": bool(permissive_recovery),
            }
        )
    return events


def _compute_reappearance_rates(batch_dir: Path) -> dict[str, Any]:
    """Compute global perfect_rate and permissive_rate across all scenes."""
    scenes_root = batch_dir / "scenes"
    if not scenes_root.is_dir():
        return {"total_reappearances": 0, "perfect_rate": None, "permissive_rate": None}

    total_reappearances = 0
    total_perfect = 0
    total_permissive = 0

    for scene_dir in sorted(
        child for child in scenes_root.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    ):
        per_object_rows = _read_csv_rows(scene_dir / "per_object.csv")
        for row in per_object_rows:
            events = _build_reappearance_events_for_object(row, scene_name=scene_dir.name)
            total_reappearances += len(events)
            total_perfect += sum(1 for e in events if e["perfect_recovery"])
            total_permissive += sum(1 for e in events if e["permissive_recovery"])

    return {
        "total_reappearances": total_reappearances,
        "total_perfect": total_perfect,
        "total_permissive": total_permissive,
        "perfect_rate": _safe_pct(total_perfect, total_reappearances),
        "permissive_rate": _safe_pct(total_permissive, total_reappearances),
    }


# ---------------------------------------------------------------------------
# Main: generate paper table row
# ---------------------------------------------------------------------------

def generate_paper_row(
    batch_dir: Path,
    *,
    method: str,
    detector: str,
) -> dict[str, Any]:
    batch_dir = batch_dir.resolve()

    # 1) Parse report.txt
    report = _parse_report(batch_dir / "report.txt")

    # 2) Parse failed_scenes.txt
    failed_scenes = _parse_failed_scenes(batch_dir / "failed_scenes.txt")
    n_completed = _report_int(report, "n_scenes_completed") or 0
    n_failed = len(failed_scenes)
    n_total = n_completed + n_failed
    failed_pct = (n_failed / n_total * 100.0) if n_total > 0 else 0.0

    # 3) n_cases for IDSW rate: from report if available, else count per_case.csv rows
    n_cases = _report_int(report, "n_cases")
    if n_cases is None:
        per_case_path = batch_dir / "per_case.csv"
        if per_case_path.is_file():
            n_cases = _count_csv_rows(per_case_path)
            print(f"  [INFO] n_cases not in report.txt, counted {n_cases} rows from per_case.csv")
        else:
            # Try aggregating from scene-level per_case.csv files
            scenes_root = batch_dir / "scenes"
            if scenes_root.is_dir():
                n_cases = 0
                for scene_dir in sorted(
                    child for child in scenes_root.iterdir()
                    if child.is_dir() and not child.name.startswith(".")
                ):
                    scene_per_case = scene_dir / "per_case.csv"
                    if scene_per_case.is_file():
                        n_cases += _count_csv_rows(scene_per_case)
                print(f"  [INFO] n_cases aggregated from scene-level per_case.csv: {n_cases}")

    idsw = _report_int(report, "idsw")
    idsw_rate = None
    if idsw is not None and n_cases and n_cases > 0:
        idsw_rate = idsw / n_cases * 100.0

    # 4) Reappearance analysis → Recovery / Hard rec.
    reapp = _compute_reappearance_rates(batch_dir)
    recovery_pct = reapp["permissive_rate"] * 100.0 if reapp["permissive_rate"] is not None else None
    hard_rec_pct = reapp["perfect_rate"] * 100.0 if reapp["perfect_rate"] is not None else None

    # 5) Metrics from report.txt
    idf1 = _report_float(report, "idf1")
    idp = _report_float(report, "idp")
    idr = _report_float(report, "idr")
    deta = _report_float(report, "deta")
    assa = _report_float(report, "assa")

    row = {
        "method": method,
        "detector": detector,
        # Table 1
        "idf1": idf1,
        "recovery": recovery_pct,
        "hard_rec": hard_rec_pct,
        "idsw_rate": idsw_rate,
        "failed_scenes_pct": failed_pct,
        # Table 2
        "deta": deta,
        "assa": assa,
        "idp": idp,
        "idr": idr,
        # Raw counts (informational)
        "n_completed": n_completed,
        "n_failed": n_failed,
        "n_cases": n_cases,
        "idsw": idsw,
        "total_reappearances": reapp["total_reappearances"],
        "total_perfect": reapp.get("total_perfect"),
        "total_permissive": reapp.get("total_permissive"),
    }
    return row


def _fmt(val: float | None, decimals: int = 2) -> str:
    if val is None:
        return "-%"
    return f"{val:.{decimals}f}%"


def _build_paper_tables_text(row: dict[str, Any]) -> str:
    m = row["method"]
    d = row["detector"]
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append(f"  {m} / {d}")
    lines.append("=" * 80)
    lines.append("")

    # Table 1
    lines.append("Table 1: Method | Detector | IDF1 | Recovery | Hard rec. | IDSW rate | Failed Scenes")
    lines.append(
        f"  {m:15s} | {d:12s} | {_fmt(row['idf1'])} | {_fmt(row['recovery'])} | "
        f"{_fmt(row['hard_rec'])} | {_fmt(row['idsw_rate'])} | {_fmt(row['failed_scenes_pct'], 1)}"
    )
    lines.append("")

    # Table 2
    lines.append("Table 2: Method | Detector | DetA | AssA | IDP | IDR | Failed Scenes")
    lines.append(
        f"  {m:15s} | {d:12s} | {_fmt(row['deta'])} | {_fmt(row['assa'])} | "
        f"{_fmt(row['idp'])} | {_fmt(row['idr'])} | {_fmt(row['failed_scenes_pct'], 1)}"
    )
    lines.append("")

    # LaTeX-ready rows
    lines.append("--- LaTeX (Table 1) ---")
    lines.append(
        f"{m} & {d} & {_fmt(row['idf1'])} & {_fmt(row['recovery'])} & "
        f"{_fmt(row['hard_rec'])} & {_fmt(row['idsw_rate'])} & {_fmt(row['failed_scenes_pct'], 1)} \\\\"
    )
    lines.append("")
    lines.append("--- LaTeX (Table 2) ---")
    lines.append(
        f"{m} & {d} & {_fmt(row['deta'])} & {_fmt(row['assa'])} & "
        f"{_fmt(row['idp'])} & {_fmt(row['idr'])} & {_fmt(row['failed_scenes_pct'], 1)} \\\\"
    )
    lines.append("")

    # Raw counts
    lines.append("--- Raw counts ---")
    lines.append(f"  n_completed:          {row['n_completed']}")
    lines.append(f"  n_failed:             {row['n_failed']}")
    lines.append(f"  n_cases:              {row['n_cases']}")
    lines.append(f"  idsw:                 {row['idsw']}")
    lines.append(f"  total_reappearances:  {row['total_reappearances']}")
    lines.append(f"  total_perfect:        {row['total_perfect']}")
    lines.append(f"  total_permissive:     {row['total_permissive']}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper table values from a batch results directory."
    )
    parser.add_argument(
        "batch_dir", type=Path,
        help="Batch results directory (e.g. .../scannetpp_d4sm_yolo_v2)",
    )
    parser.add_argument("--method", type=str, required=True, help="Method name for the table (e.g. REMIND, DAM4SAM)")
    parser.add_argument("--detector", type=str, required=True, help="Detector label (e.g. 'GT masks', 'YOLO')")
    args = parser.parse_args()

    batch_dir = args.batch_dir.resolve()
    row = generate_paper_row(
        batch_dir,
        method=args.method,
        detector=args.detector,
    )

    text = _build_paper_tables_text(row)
    print(text)

    out_path = batch_dir / "tables_paper.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"[OK] Written to {out_path}")


if __name__ == "__main__":
    main()