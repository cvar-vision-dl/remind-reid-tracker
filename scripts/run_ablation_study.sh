#!/usr/bin/env bash
set -uo pipefail
# NOTE: no -e so the script continues after a scene fails / is OOM-killed.

# =============================================================================
# REMIND Ablation Study
#   - 3 experiments on ScanNet++ (no_bg, no_neighbor, greedy)
#   - 3 experiments on custom video (no_bg, no_neighbor, greedy)
# All using GT masks (no YOLO).
#
# Each scene runs as a separate Python process so that an OOM kill (SIGKILL)
# only takes down that scene — the shell script survives, records the failure,
# and moves on to the next scene.
# =============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TESTING_DIR="${REPO_ROOT}/testing"

# Paths
SCANNET_DATASET_ROOT="/mnt/a/alejodosr/qsync/2026_tracker_reid/datasets/scannetpp_data/"
SCANNET_ANNOTATIONS_TAR_DIR="${SCANNET_DATASET_ROOT}/annotations"
CUSTOM_FRAMES_DIR="/mnt/a/alejodosr/qsync/2026_tracker_reid/datasets/custom_video/FRAMES/"
CUSTOM_META="/mnt/a/alejodosr/qsync/2026_tracker_reid/datasets/custom_video/DAVIS_OUT/metaCUSTOMVIDEO.json"
CUSTOM_ANNOTATIONS="/mnt/a/alejodosr/qsync/2026_tracker_reid/datasets/custom_video/DAVIS_OUT/Annotations/raw/FRAMES/"
OUTPUT_BASE="/mnt/a/alejodosr/sherec/reid_tracker/outputs/paper/ablation"

# Config files (full copies of default with one change each)
CFG_NO_BG="${REPO_ROOT}/config/ablation_no_bg.yaml"
CFG_NO_NEIGHBOR="${REPO_ROOT}/config/ablation_no_neighbor.yaml"
CFG_GREEDY="${REPO_ROOT}/config/ablation_greedy.yaml"

# Ablation definitions: name / config path
ABLATIONS=(
    "no_bg:${CFG_NO_BG}"
    "no_neighbor:${CFG_NO_NEIGHBOR}"
    "greedy:${CFG_GREEDY}"
)

# ---------------------------------------------------------------------------
# Discover ScanNet++ scene IDs from .tar files in the data directory.
# ---------------------------------------------------------------------------
discover_scannet_scenes() {
    local data_dir="$1"
    local scenes=()
    for tar in "${data_dir}"/*.tar; do
        [ -f "$tar" ] || continue
        scenes+=("$(basename "$tar" .tar)")
    done
    # Sort for deterministic order
    printf '%s\n' "${scenes[@]}" | sort
}

# ---------------------------------------------------------------------------
# Describe exit code for logging.
# ---------------------------------------------------------------------------
exit_code_label() {
    local code="$1"
    if [ "$code" -eq 137 ]; then
        echo "OOM/SIGKILL (exit 137)"
    elif [ "$code" -eq 139 ]; then
        echo "SIGSEGV (exit 139)"
    else
        echo "exit $code"
    fi
}

echo "================================================================"
echo " REMIND Ablation Study"
echo " Output base: ${OUTPUT_BASE}"
echo "================================================================"

# Run from the testing directory so relative imports resolve correctly
cd "${TESTING_DIR}"

# -------------------------------------------------------------------------
# ScanNet++ ablations — one process per scene
# -------------------------------------------------------------------------
SCANNET_SCENES=( $(discover_scannet_scenes "${SCANNET_ANNOTATIONS_TAR_DIR}") )
echo ""
echo " Discovered ${#SCANNET_SCENES[@]} ScanNet++ scenes."

for entry in "${ABLATIONS[@]}"; do
    name="${entry%%:*}"
    cfg="${entry#*:}"
    run_id="ablation_scannet_remind_${name}"
    output_dir="${OUTPUT_BASE}/${run_id}"
    failed_file="${output_dir}/failed_scenes.txt"

    echo ""
    echo "================================================================"
    echo " [ScanNet++] ${run_id}"
    echo "   config:  ${cfg}"
    echo "   output:  ${output_dir}"
    echo "   scenes:  ${#SCANNET_SCENES[@]}"
    echo "================================================================"

    mkdir -p "${output_dir}"

    completed=0
    failed=0
    skipped=0
    failed_scenes=()

    # Files that must all exist for a scene to be considered complete
    COMPLETE_FILES=(scene_summary.csv per_class.csv per_object.csv per_case.csv per_case_modules.csv per_frame.csv per_pred_track.csv per_event.csv)

    for scene_id in "${SCANNET_SCENES[@]}"; do
        echo ""
        echo "--- [ScanNet++] ${run_id} | scene ${scene_id} ---"

        # Fast shell-level skip: check if all output files already exist
        scene_dir="${output_dir}/scenes/${scene_id}"
        all_present=true
        for f in "${COMPLETE_FILES[@]}"; do
            if [ ! -f "${scene_dir}/${f}" ]; then
                all_present=false
                break
            fi
        done
        if [ "$all_present" = true ]; then
            echo "[SHELL] Skip completed scene -> ${scene_id}"
            completed=$((completed + 1))
            continue
        fi

        python run_tracking_batch_tar.py \
            --dataset-root "${SCANNET_DATASET_ROOT}" \
            --config-path "${cfg}" \
            --output-dir "${output_dir}" \
            --run-id "${run_id}" \
            --scene-id "${scene_id}"
        rc=$?

        if [ "$rc" -eq 0 ]; then
            completed=$((completed + 1))
        else
            failed=$((failed + 1))
            label="$(exit_code_label $rc)"
            echo "[ScanNet++][FAILED] ${scene_id} — ${label}"
            failed_scenes+=("${scene_id}")
        fi
    done

    echo ""
    echo "[ScanNet++] ${run_id} done: completed=${completed} failed=${failed}"
    if [ "${#failed_scenes[@]}" -gt 0 ]; then
        IFS=','; echo "${failed_scenes[*]}" > "${failed_file}"; unset IFS
        echo "[ScanNet++] Failed scenes written to: ${failed_file}"
    fi
done

# -------------------------------------------------------------------------
# Custom video ablations
# -------------------------------------------------------------------------
for entry in "${ABLATIONS[@]}"; do
    name="${entry%%:*}"
    cfg="${entry#*:}"
    run_id="ablation_custom_remind_${name}"
    output_dir="${OUTPUT_BASE}/${run_id}"
    failed_file="${output_dir}/failed_scenes.txt"

    echo ""
    echo "================================================================"
    echo " [Custom Video] ${run_id}"
    echo "   config: ${cfg}"
    echo "   output: ${output_dir}"
    echo "================================================================"

    mkdir -p "${output_dir}"

    python run_tracking_batch_custom.py \
        --frames-dir "${CUSTOM_FRAMES_DIR}" \
        --davis-meta-path "${CUSTOM_META}" \
        --davis-annotations-dir "${CUSTOM_ANNOTATIONS}" \
        --config-path "${cfg}" \
        --output-dir "${output_dir}" \
        --run-id "${run_id}"
    rc=$?

    if [ "$rc" -eq 0 ]; then
        echo "[Custom Video] ${run_id} finished successfully."
    else
        label="$(exit_code_label $rc)"
        echo "[Custom Video][FAILED] ${run_id} — ${label}"
        echo "CUSTOMVIDEO" > "${failed_file}"
    fi
done

echo ""
echo "================================================================"
echo " All ablation experiments completed."
echo "================================================================"
