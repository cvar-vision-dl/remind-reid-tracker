from __future__ import annotations

from utils.visualization import (
    draw_dino_patch_grid,
    draw_obj_and_bg_rings_from_det_features,
    draw_track_labels_from_update_output,
)


def build_association_frame_preview(
    *,
    config: dict,
    memory_store,
    patch_size: int,
    frame_bgr,
    perception_output,
    association_output,
    update_output,
):
    if frame_bgr is None:
        return None

    dbg_cfg = (config.get("debug", {}) or {})
    viz_cfg = (dbg_cfg.get("visualization", {}) or {})

    frame_aligned = ((getattr(perception_output, "debug", None) or {}).get("frame_aligned_bgr", None))
    viz = frame_bgr.copy() if frame_aligned is None else frame_aligned.copy()

    detections = getattr(perception_output, "detections", None) or []
    det_features_by_id = getattr(perception_output, "det_features_by_id", None) or {}

    if bool(viz_cfg.get("show_bg_rings", True)) or bool(viz_cfg.get("show_obj_mask", True)):
        rings_cfg = (viz_cfg.get("rings", {}) or {})
        obj_cfg = (viz_cfg.get("object_mask", {}) or {})
        viz = draw_obj_and_bg_rings_from_det_features(
            frame_bgr=viz,
            detections=detections,
            det_features_by_id=det_features_by_id,
            patch_size=int(patch_size),
            update_output=update_output,
            memory_store=memory_store,
            color_mode=str(obj_cfg.get("color_mode", "davis")),
            alpha_inner=float(rings_cfg.get("alpha_inner", 0.35)),
            alpha_outer=float(rings_cfg.get("alpha_outer", 0.25)),
            outer_dark_factor=float(rings_cfg.get("outer_dark_factor", 0.65)),
            show_bg_rings=bool(viz_cfg.get("show_bg_rings", True)),
            show_obj_mask=bool(viz_cfg.get("show_obj_mask", True)),
            alpha_obj=float(obj_cfg.get("alpha", 0.55)),
            obj_brighten_add=int(obj_cfg.get("brighten_add", 55)),
            draw_obj_contour=bool(obj_cfg.get("draw_contour", True)),
            contour_thickness=int(obj_cfg.get("contour_thickness", 2)),
        )

    if bool(viz_cfg.get("show_track_labels", True)):
        viz = draw_track_labels_from_update_output(
            frame_bgr=viz,
            detections=detections,
            update_output=update_output,
            memory_store=memory_store,
            assoc_output=association_output,
            show_score=bool(viz_cfg.get("show_track_score", False)),
            show_kind=bool(viz_cfg.get("show_track_kind", False)),
            draw_bg_box=bool(viz_cfg.get("track_label_draw_bg_box", True)),
            color_mode=str(viz_cfg.get("track_label_color_mode", "davis")),
            avoid_masks=bool(viz_cfg.get("track_label_avoid_masks", True)),
            allow_on_large_masks=bool(viz_cfg.get("track_label_allow_on_large_masks", True)),
            large_mask_area_frac=float(viz_cfg.get("track_label_large_mask_area_frac", 0.08)),
            draw_link_line=bool(viz_cfg.get("track_label_draw_link_line", True)),
            link_min_dist_px=int(viz_cfg.get("track_label_link_min_dist_px", 28)),
            link_thickness=int(viz_cfg.get("track_label_link_thickness", 1)),
            sticky=bool(viz_cfg.get("track_label_sticky", True)),
            sticky_max_dist_px=int(viz_cfg.get("track_label_sticky_max_dist_px", 220)),
        )

    if bool(viz_cfg.get("show_dino_grid", False)):
        grid_cfg = (viz_cfg.get("dino_grid", {}) or {})
        viz = draw_dino_patch_grid(
            frame_bgr=viz,
            patch_size=int(patch_size),
            enabled=True,
            step_patches=int(grid_cfg.get("step_patches", 1)),
            thickness=int(grid_cfg.get("thickness", 1)),
            alpha=float(grid_cfg.get("alpha", 0.25)),
        )

    return viz
