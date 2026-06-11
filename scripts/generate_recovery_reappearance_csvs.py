from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import Any


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


def _format_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    return value


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key)) for key in fieldnames})


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

    visibility_segments = _split_visibility_segments([int(frame) for frame in frames_timeline if frame is not None])
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
                "gt_label": row.get("gt_label", ""),
                "gt_class_name": row.get("gt_class_name", ""),
                "reference_pred_id": reference_pred_id,
                "reappearance_index": int(reappearance_index),
                "start_frame": int(segment_frames[0]),
                "end_frame": int(segment_frames[-1]),
                "length_visible_frames": int(len(segment_frames)),
                "first_pred_id": first_pred_id,
                "first_ok_frame": first_ok_frame,
                "perfect_recovery": bool(perfect_recovery),
                "permissive_recovery": bool(permissive_recovery),
            }
        )
    return events


def _build_scene_outputs(scene_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_object_rows = _read_csv_rows(scene_dir / "per_object.csv")
    if not per_object_rows:
        return [], []

    scene_name = str(scene_dir.name)

    event_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    total_reappearances = 0
    total_perfect_hits = 0
    total_permissive_hits = 0

    for row in per_object_rows:
        gt_events = _build_reappearance_events_for_object(
            row,
            scene_name=scene_name,
        )
        event_rows.extend(gt_events)

        reappearances = int(len(gt_events))
        perfect_hits = int(sum(1 for event in gt_events if bool(event["perfect_recovery"])))
        permissive_hits = int(sum(1 for event in gt_events if bool(event["permissive_recovery"])))
        total_reappearances += reappearances
        total_perfect_hits += perfect_hits
        total_permissive_hits += permissive_hits

        summary_rows.append(
            {
                "scene_name": scene_name,
                "gt_instance_id": _parse_optional_int(row.get("gt_instance_id")),
                "gt_label": row.get("gt_label", ""),
                "gt_class_name": row.get("gt_class_name", ""),
                "reappearances": reappearances,
                "perfect_hits": perfect_hits,
                "perfect_rate": _safe_pct(perfect_hits, reappearances),
                "permissive_hits": permissive_hits,
                "permissive_rate": _safe_pct(permissive_hits, reappearances),
            }
        )

    global_row = {
        "scene_name": scene_name,
        "gt_instance_id": "GLOBAL",
        "gt_label": "GLOBAL",
        "gt_class_name": "GLOBAL",
        "reappearances": int(total_reappearances),
        "perfect_hits": int(total_perfect_hits),
        "perfect_rate": _safe_pct(total_perfect_hits, total_reappearances),
        "permissive_hits": int(total_permissive_hits),
        "permissive_rate": _safe_pct(total_permissive_hits, total_reappearances),
    }
    return [global_row, *summary_rows], event_rows


def _infer_default_tags(batch_dir: Path) -> tuple[str, str]:
    dataset = batch_dir.name
    model = batch_dir.parent.name
    return model, dataset


def generate_outputs(batch_dir: Path, *, model: str, dataset: str) -> dict[str, Any]:
    scenes_root = batch_dir / "scenes"
    if not scenes_root.is_dir():
        raise FileNotFoundError(f"No existe scenes root: {scenes_root}")

    derived_root = batch_dir / "DatosDerivados"
    scene_summary_fieldnames = [
        "scene_name",
        "gt_instance_id",
        "gt_label",
        "gt_class_name",
        "reappearances",
        "perfect_hits",
        "perfect_rate",
        "permissive_hits",
        "permissive_rate",
    ]
    scene_events_fieldnames = [
        "scene_name",
        "gt_instance_id",
        "gt_label",
        "gt_class_name",
        "reference_pred_id",
        "reappearance_index",
        "start_frame",
        "end_frame",
        "length_visible_frames",
        "first_pred_id",
        "first_ok_frame",
        "perfect_recovery",
        "permissive_recovery",
    ]
    batch_summary_fieldnames = [
        "model",
        "dataset",
        "scene_name",
        "gt_instance_id",
        "gt_label",
        "gt_class_name",
        "reappearances",
        "perfect_hits",
        "perfect_rate",
        "permissive_hits",
        "permissive_rate",
    ]
    batch_events_fieldnames = [
        "model",
        "dataset",
        "scene_name",
        "gt_instance_id",
        "gt_label",
        "gt_class_name",
        "reference_pred_id",
        "reappearance_index",
        "start_frame",
        "end_frame",
        "length_visible_frames",
        "first_pred_id",
        "first_ok_frame",
        "perfect_recovery",
        "permissive_recovery",
    ]

    all_batch_summary_rows: list[dict[str, Any]] = []
    all_batch_event_rows: list[dict[str, Any]] = []
    total_reappearances = 0
    total_perfect_hits = 0
    total_permissive_hits = 0
    scenes_processed = 0

    for scene_dir in sorted(child for child in scenes_root.iterdir() if child.is_dir() and not child.name.startswith(".")):
        scene_summary_rows, scene_event_rows = _build_scene_outputs(scene_dir)
        if not scene_summary_rows:
            continue
        scenes_processed += 1

        scene_out_dir = derived_root / "scenes" / scene_dir.name
        _write_csv_rows(scene_out_dir / "recovery_reappearance_summary.csv", scene_summary_fieldnames, scene_summary_rows)
        _write_csv_rows(scene_out_dir / "recovery_reappearance_events.csv", scene_events_fieldnames, scene_event_rows)

        scene_object_rows = scene_summary_rows[1:]
        all_batch_summary_rows.extend(
            {
                "model": model,
                "dataset": dataset,
                **row,
            }
            for row in scene_object_rows
        )
        all_batch_event_rows.extend(
            {
                "model": model,
                "dataset": dataset,
                **row,
            }
            for row in scene_event_rows
        )

        scene_global = scene_summary_rows[0]
        total_reappearances += int(scene_global["reappearances"])
        total_perfect_hits += int(scene_global["perfect_hits"])
        total_permissive_hits += int(scene_global["permissive_hits"])

    batch_global_row = {
        "model": model,
        "dataset": dataset,
        "scene_name": "GLOBAL",
        "gt_instance_id": "GLOBAL",
        "gt_label": "GLOBAL",
        "gt_class_name": "GLOBAL",
        "reappearances": int(total_reappearances),
        "perfect_hits": int(total_perfect_hits),
        "perfect_rate": _safe_pct(total_perfect_hits, total_reappearances),
        "permissive_hits": int(total_permissive_hits),
        "permissive_rate": _safe_pct(total_permissive_hits, total_reappearances),
    }
    _write_csv_rows(
        derived_root / "recovery_reappearance_summary.csv",
        batch_summary_fieldnames,
        [batch_global_row, *all_batch_summary_rows],
    )
    _write_csv_rows(
        derived_root / "recovery_reappearance_events.csv",
        batch_events_fieldnames,
        all_batch_event_rows,
    )

    return {
        "batch_dir": str(batch_dir),
        "model": model,
        "dataset": dataset,
        "scenes_processed": scenes_processed,
        "total_reappearances": total_reappearances,
        "total_perfect_hits": total_perfect_hits,
        "total_permissive_hits": total_permissive_hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenera recovery_reappearance_summary.csv y "
            "recovery_reappearance_events.csv a partir de per_object.csv."
        )
    )
    parser.add_argument("batch_dir", type=Path, help="Directorio del batch de resultados, por ejemplo Resultados/Resultados/tfm/scannetpp")
    parser.add_argument("--model", type=str, default=None, help="Etiqueta de modelo para el CSV global")
    parser.add_argument("--dataset", type=str, default=None, help="Etiqueta de dataset para el CSV global")
    args = parser.parse_args()

    batch_dir = args.batch_dir.resolve()
    default_model, default_dataset = _infer_default_tags(batch_dir)
    result = generate_outputs(
        batch_dir,
        model=args.model or default_model,
        dataset=args.dataset or default_dataset,
    )

    print(
        "[RECOVERY][REAPPEARANCE] "
        f"scenes={result['scenes_processed']} "
        f"reappearances={result['total_reappearances']} "
        f"perfect_hits={result['total_perfect_hits']} "
        f"permissive_hits={result['total_permissive_hits']}"
    )
    print(f"[RECOVERY][REAPPEARANCE] Wrote outputs under {batch_dir / 'DatosDerivados'}")


if __name__ == "__main__":
    main()
