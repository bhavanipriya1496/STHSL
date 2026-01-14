#!/usr/bin/env bash
set -euo pipefail

# ---------------- Defaults ----------------
ARCHS_ALL=("baseline" "option1" "option2")
DATASETS_ALL=("NYC" "CHI")
TARGETS_ALL=("Logs" "tests_validation" "Save")

LOGS_DIR="Logs"
TESTS_DIR="tests_validation"
SAVE_DIR="Save"

DRY_RUN=false
YES=false

# ---------------- Helpers ----------------
usage() {
  cat <<'EOF'
Usage:
  ./cleanup_experiments.sh [options]

Options:
  --dataset NYC|CHI|all                     (default: all)
  --arch baseline|option1|option2|all       (default: all)
  --target Logs|tests_validation|Save|all   (default: all)
  --dry-run                                 Print what would be deleted (non-destructive)
  -y, --yes                                 No prompt (non-interactive)
  -h, --help                                Show this help

Delete rules:
  Logs             -> delete ALL contents inside Logs/<arch>/<dataset>   (keep directory)
  tests_validation -> delete ALL contents inside tests_validation/<arch>/<dataset> (keep directory)
  Save             -> delete ONLY items matching _epoch* inside Save/<dataset> (keep directory)

Examples:
  ./cleanup_experiments.sh --dataset NYC --arch option1 --target Logs
  ./cleanup_experiments.sh --dataset all --arch baseline --target all
  ./cleanup_experiments.sh --dataset CHI --arch all --target tests_validation --dry-run
  ./cleanup_experiments.sh --dataset NYC --arch option2 --target all -y
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

# Delete ALL contents of a directory, but NOT the directory itself.
# Removes files and subdirectories, including dotfiles.
rm_all_inside() {
  local dir="$1"

  if [[ ! -d "$dir" ]]; then
    echo "SKIP (not a dir): $dir"
    return 0
  fi

  shopt -s dotglob nullglob
  local items=("$dir"/*)
  shopt -u dotglob

  if (( ${#items[@]} == 0 )); then
    echo "EMPTY: $dir"
    return 0
  fi

  if [[ "$DRY_RUN" == true ]]; then
    for it in "${items[@]}"; do
      echo "DRY-RUN: rm -rf \"${it}\""
    done
  else
    echo "CLEAN (all contents): $dir"
    rm -rf -- "${items[@]}"
  fi
}

# Delete ONLY top-level items in dir matching _epoch*
# (does NOT recurse into subdirectories unless those subdirs themselves match _epoch*)
rm_epoch_inside() {
  local dir="$1"

  if [[ ! -d "$dir" ]]; then
    echo "SKIP (not a dir): $dir"
    return 0
  fi

  shopt -s nullglob
  local matches=("$dir"/_epoch*)
  shopt -u nullglob

  if (( ${#matches[@]} == 0 )); then
    echo "NO-MATCH: $dir/_epoch*"
    return 0
  fi

  if [[ "$DRY_RUN" == true ]]; then
    for it in "${matches[@]}"; do
      echo "DRY-RUN: rm -rf \"${it}\""
    done
  else
    echo "CLEAN (pattern _epoch*): $dir"
    rm -rf -- "${matches[@]}"
  fi
}

# ---------------- Parse args ----------------
DATASET="all"
ARCH="all"
TARGET="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --arch)    ARCH="${2:-}"; shift 2 ;;
    --target)  TARGET="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -y|--yes)  YES=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

# Validate
[[ "$DATASET" == "all" || "$DATASET" == "NYC" || "$DATASET" == "CHI" ]] || die "--dataset must be NYC|CHI|all"
[[ "$ARCH" == "all" || "$ARCH" == "baseline" || "$ARCH" == "option1" || "$ARCH" == "option2" ]] || die "--arch must be baseline|option1|option2|all"
[[ "$TARGET" == "all" || "$TARGET" == "Logs" || "$TARGET" == "tests_validation" || "$TARGET" == "Save" ]] || die "--target must be Logs|tests_validation|Save|all"

# Expand sets
if [[ "$DATASET" == "all" ]]; then DATASETS=("${DATASETS_ALL[@]}"); else DATASETS=("$DATASET"); fi
if [[ "$ARCH" == "all" ]]; then ARCHS=("${ARCHS_ALL[@]}"); else ARCHS=("$ARCH"); fi
if [[ "$TARGET" == "all" ]]; then TARGETS=("${TARGETS_ALL[@]}"); else TARGETS=("$TARGET"); fi

# ---------------- Prompt ----------------
if [[ "$YES" == false ]]; then
  echo "About to clean (directories kept):"
  echo "  Datasets : ${DATASETS[*]}"
  echo "  Archs    : ${ARCHS[*]}"
  echo "  Targets  : ${TARGETS[*]}"
  echo "  Dry-run  : $DRY_RUN"
  echo
  echo "Rules:"
  echo "  Logs             -> delete ALL contents inside Logs/<arch>/<dataset>"
  echo "  tests_validation -> delete ALL contents inside tests_validation/<arch>/<dataset>"
  echo "  Save             -> delete ONLY _epoch* inside Save/<dataset>"
  echo

  confirm=""
  # IMPORTANT: read from TTY (fix for Git Bash / MINGW64)
  if ! read -r -p "Continue? [y/N]: " confirm </dev/tty; then
    echo "Could not read confirmation from terminal. Aborted."
    exit 1
  fi
  [[ "$confirm" == "y" || "$confirm" == "Y" ]] || { echo "Aborted."; exit 0; }
fi

# ---------------- Clean ----------------
for t in "${TARGETS[@]}"; do
  case "$t" in
    "Logs")
      for a in "${ARCHS[@]}"; do
        for d in "${DATASETS[@]}"; do
          rm_all_inside "${LOGS_DIR}/${a}/${d}"
        done
      done
      ;;
    "tests_validation")
      for a in "${ARCHS[@]}"; do
        for d in "${DATASETS[@]}"; do
          rm_all_inside "${TESTS_DIR}/${a}/${d}"
        done
      done
      ;;
    "Save")
      for d in "${DATASETS[@]}"; do
        rm_epoch_inside "${SAVE_DIR}/${d}"
      done
      ;;
  esac
done

echo "Cleanup done."
