#!/usr/bin/env bash
# Every model, every stain: 5-fold CV on training slides (seed 42) + reference slides (seeds 42-44).
# Resumable: a run whose log already has a RESULT line is skipped. Run from LEMON/:
#   nohup bash set_transformers/run_compare.sh > runs/compare.out 2>&1 &
# Subsets:  STAINS="Ki67" CONFIGS="mlp mlp_norm" PHASES="cv" bash set_transformers/run_compare.sh
set -u
STAINS=${STAINS:-"Ki67 ER PR"}
PHASES=${PHASES:-"cv ref"}
FOLDS=${FOLDS:-5}
REF_SEEDS=${REF_SEEDS:-"42 43 44"}
OUT=${OUT:-runs/compare}
EXTRA=${EXTRA:-}          # e.g. EXTRA="--epochs 2" for a quick check
M=train.py
T=set_transformers/train_partition.py
F="--input fusion --lemon_pca 64 --no_labelled"   # labelled-binary nuclei have no LEMON embeddings
N="--slide_norm center --slide_norm_stats all"
declare -A CMD=(
  [mlp]="$M --input qupath"
  [mlp_norm]="$M --input qupath $N"
  [set1]="$T --set_size 1 --batch_sets 3072"
  [set32]="$T --set_size 32 --batch_sets 96"
  [shuf48]="$T --set_size 48 --batch_sets 64 --shuffle_sets"
  [set1_norm]="$T --set_size 1 --batch_sets 3072 $N"
  [set32_norm]="$T --set_size 32 --batch_sets 96 $N"
  [shuf48_norm]="$T --set_size 48 --batch_sets 64 --shuffle_sets $N"
  [fus_mlp]="$M $F"
  [fus_mlp_norm]="$M $F $N"
  [fus_set32]="$T --set_size 32 --batch_sets 96 $F"
  [fus_set32_norm]="$T --set_size 32 --batch_sets 96 $F $N"
)
# Nuc2Vec / LEMON-only configs (run with e.g. CONFIGS="n2v_mlp n2v_fus_mlp_norm n2v_shuf48 n2v_fus_shuf48_norm").
#   n2v_*   one encoder pretrained on all stains  -> embeddings/nuc2vec_pooled.npz
#   n2vs_*  stain-specific encoders               -> embeddings/nuc2vec_<stain>.npz
#   n2vm_*  pooled encoder pretrained with nucleus MASKS (bg_mode mask) -> $EMB_MASK (default embeddings/nuc2vec_mask.npz)
# Nuc2Vec embeddings cover the labelled-binary nuclei when data/nuclei_manifest.csv was built without --no_labelled.
declare -A BASE=(
  [mlp]="$M --input lemon --lemon @E@"
  [fus_mlp]="$M --input fusion --lemon @E@"
  [fus_mlp_norm]="$M --input fusion --lemon @E@ $N"
  [shuf48]="$T --input lemon --lemon @E@ --set_size 48 --batch_sets 64 --shuffle_sets"
  [fus_shuf48_norm]="$T --input fusion --lemon @E@ --set_size 48 --batch_sets 64 --shuffle_sets $N"
)
for k in "${!BASE[@]}"; do
  CMD[n2v_$k]=${BASE[$k]//@E@/@EMB@}
  CMD[n2vs_$k]=${BASE[$k]//@E@/@EMBS@}
  CMD[n2vm_$k]=${BASE[$k]//@E@/@EMBM@}
done
CMD[lem_mlp]="$M --input lemon --no_labelled"                       # LEMON embeddings alone (no QuPath features)
CMD[lem_shuf48]="$T --input lemon --set_size 48 --batch_sets 64 --shuffle_sets --no_labelled"
EMB_POOLED=${EMB_POOLED:-embeddings/nuc2vec_pooled.npz}
EMB_MASK=${EMB_MASK:-embeddings/nuc2vec_mask.npz}
CONFIGS=${CONFIGS:-"mlp mlp_norm set1 set32 shuf48 set1_norm set32_norm shuf48_norm fus_mlp fus_mlp_norm fus_set32 fus_set32_norm"}

run() {  # log, args...
  local log=$1; shift
  if [ -f "$log" ] && grep -q '^RESULT' "$log"; then return; fi
  # another runner is probably on this one right now (log written to in the last 10 minutes): do not clobber it
  if [ -f "$log" ] && [ -n "$(find "$log" -mmin -10 2>/dev/null)" ]; then echo "    in progress elsewhere, skipping: $log"; return; fi
  mkdir -p "$(dirname "$log")"
  echo "$(date '+%F %T')  $log"
  python3 "$@" > "$log" 2>&1 || echo "    FAILED: tail $log"
}

for st in $STAINS; do
  for cfg in $CONFIGS; do
    c=${CMD[$cfg]:-}; [ -z "$c" ] && { echo "unknown config $cfg"; continue; }
    case $cfg in   # one clear message instead of dozens of instantly failing runs
      n2v_*)  [ -f "$EMB_POOLED" ] || { echo "missing embeddings file $EMB_POOLED - skipping $cfg ($st)"; continue; } ;;
      n2vm_*) [ -f "$EMB_MASK" ] || { echo "missing embeddings file $EMB_MASK - skipping $cfg ($st)"; continue; } ;;
      n2vs_*) [ -f "embeddings/nuc2vec_${st}.npz" ] || { echo "missing embeddings file embeddings/nuc2vec_${st}.npz - skipping $cfg"; continue; } ;;
    esac
    c=${c//@EMBS@/embeddings/nuc2vec_${st}.npz}; c=${c//@EMBM@/$EMB_MASK}; c=${c//@EMB@/$EMB_POOLED}
    for ph in $PHASES; do
      if [ "$ph" = cv ]; then
        for f in $(seq 0 $((FOLDS - 1))); do
          run "$OUT/$st/cv/${cfg}_f$f.log" $c $EXTRA --subtypes $st --seed 42 --cv_folds $FOLDS --fold $f --skip_test
        done
      else
        for s in $REF_SEEDS; do
          run "$OUT/$st/ref/${cfg}_s$s.log" $c $EXTRA --subtypes $st --seed $s
        done
      fi
    done
  done
done
echo "$(date '+%F %T')  done"