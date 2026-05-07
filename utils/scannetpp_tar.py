from __future__ import annotations

import json
import os
import shutil
import tarfile
from pathlib import Path


def resolve_scannetpp_dataset_root(base_path: Path) -> Path | None:
    base = base_path.expanduser().resolve()
    if (base / "data").is_dir():
        return base
    if base.name == "data" and base.parent.is_dir():
        return base.parent.resolve()
    return None


def resolve_scannetpp_data_parent(base_path: Path) -> Path | None:
    base = base_path.expanduser().resolve()
    if base.name == "data" and base.is_dir():
        return base
    candidate = (base / "data").resolve()
    if candidate.is_dir():
        return candidate
    return None


def resolve_scannetpp_tar_cache_root(project_dir: str | Path) -> Path:
    raw = str(Path(project_dir).resolve() / ".cache" / "scannetpp_tar")
    override = str(os.environ.get("APP2_SCANNETPP_TAR_CACHE_ROOT", raw) or raw).strip()
    return Path(override).expanduser().resolve()


def resolve_scene_tar_path(*, images_root_base: Path, scene_id: str) -> Path | None:
    scene_name = str(scene_id or "").strip()
    if not scene_name:
        return None
    data_parent = resolve_scannetpp_data_parent(images_root_base)
    candidates: list[Path] = []
    if data_parent is not None:
        candidates.append((data_parent / f"{scene_name}.tar").resolve())
    candidates.append((images_root_base.expanduser().resolve() / f"{scene_name}.tar").resolve())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_scene_annotations_tar_path(*, images_root_base: Path, scene_id: str) -> Path | None:
    scene_name = str(scene_id or "").strip()
    if not scene_name:
        return None
    dataset_root = resolve_scannetpp_dataset_root(images_root_base)
    candidates: list[Path] = []
    if dataset_root is not None:
        candidates.append((dataset_root / "annotations" / f"{scene_name}.tar").resolve())
    candidates.append((images_root_base.expanduser().resolve() / "annotations" / f"{scene_name}.tar").resolve())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _safe_extract_tar(*, tar_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            member_name = str(member.name or "")
            if member_name.startswith("/") or ".." in Path(member_name).parts:
                raise RuntimeError(f"Unsafe tar member in {tar_path}: {member_name}")
        tf.extractall(dest_dir)

    scene_dir = (dest_dir / tar_path.stem).resolve()
    if scene_dir.is_dir():
        return scene_dir

    children = [p.resolve() for p in dest_dir.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]
    raise RuntimeError(
        f"No se pudo localizar la escena extraida desde {tar_path}. "
        f"Esperaba {tar_path.stem}/ y encontré {[child.name for child in children]}"
    )


def ensure_scene_tar_extracted(
    *,
    tar_path: Path,
    cache_root: Path,
    marker_name: str = "data",
    clear_existing: bool = True,
) -> Path:
    scene_id = tar_path.stem
    scene_cache_dir = (cache_root / scene_id).resolve()
    scene_root = (scene_cache_dir / scene_id).resolve()
    safe_marker_name = str(marker_name or "data").strip().lower() or "data"
    ready_marker = (scene_cache_dir / f".extracted_from_tar_{safe_marker_name}.json").resolve()

    tar_stat = tar_path.stat()
    tar_signature = {
        "tar_path": str(tar_path.resolve()),
        "size": int(tar_stat.st_size),
        "mtime_ns": int(tar_stat.st_mtime_ns),
    }

    marker_payload: dict | None = None
    if ready_marker.is_file():
        try:
            marker_payload = json.loads(ready_marker.read_text(encoding="utf-8")) or {}
        except Exception:
            marker_payload = None

    if marker_payload == tar_signature and scene_root.is_dir():
        return scene_root

    if clear_existing and scene_cache_dir.exists():
        shutil.rmtree(scene_cache_dir)
    scene_cache_dir.mkdir(parents=True, exist_ok=True)

    extracted_scene_dir = _safe_extract_tar(tar_path=tar_path, dest_dir=scene_cache_dir)
    ready_marker.write_text(
        json.dumps(tar_signature, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return extracted_scene_dir


def resolve_prepared_scene_from_tar(
    *,
    project_dir: str | Path,
    images_root_base: Path,
    scene_id: str,
    mask_variant: str,
    image_subdir: str,
) -> dict[str, str] | None:
    tar_path = resolve_scene_tar_path(images_root_base=images_root_base, scene_id=scene_id)
    if tar_path is None:
        return None
    annotations_tar_path = resolve_scene_annotations_tar_path(
        images_root_base=images_root_base,
        scene_id=scene_id,
    )

    cache_root = resolve_scannetpp_tar_cache_root(project_dir)
    scene_root = ensure_scene_tar_extracted(
        tar_path=tar_path,
        cache_root=cache_root,
        marker_name="data",
        clear_existing=True,
    )
    if annotations_tar_path is not None:
        ensure_scene_tar_extracted(
            tar_path=annotations_tar_path,
            cache_root=cache_root,
            marker_name="annotations",
            clear_existing=False,
        )

    variant = str(mask_variant or "").strip().lower() or "benchmark"
    if variant == "benchmark":
        variant = "benchmark_instance"

    frames_dir = (scene_root / image_subdir).resolve()
    meta_path = (scene_root / f"meta_{variant}.json").resolve()
    annotations_dir = (scene_root / "annotations" / variant).resolve()

    missing: list[str] = []
    if not frames_dir.is_dir():
        missing.append(f"frames:{frames_dir}")
    if not meta_path.is_file():
        missing.append(f"meta:{meta_path}")
    if not annotations_dir.is_dir():
        missing.append(f"annotations:{annotations_dir}")

    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"La escena {scene_id} existe como tar ({tar_path}), pero no está preparada "
            f"para APP2 todavía. Faltan: {missing_text}. "
            f"Si usas tars separados, asegúrate de tener también "
            f"{resolve_scene_annotations_tar_path(images_root_base=images_root_base, scene_id=scene_id) or '<dataset>/annotations/<scene>.tar'}. "
            f"Si no existe, prepara el tar con scripts/datasets/scannetpp/export_scannetpp_to_davis_tar.py "
            f"(modo in-place o salida a annotations.tar) para generar meta + annotations."
        )

    return {
        "mode": "external_scannetpp_tar",
        "frames_dir": str(frames_dir),
        "sequence_name": str(scene_id),
        "davis_meta_path": str(meta_path),
        "davis_annotations_dir": str(annotations_dir),
        "image_subdir": str(image_subdir),
        "scene_tar_path": str(tar_path),
        "scene_root": str(scene_root),
    }
