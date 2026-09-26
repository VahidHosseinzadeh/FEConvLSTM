#!/bin/bash
# Leave-subjects-out evaluation of common-fate Weizmann classification: one job per
# (held-out subject, model). The next subject in the list is the validation subject.
#
#   bash motion_benchmarks/slurm/cf_weizmann_folds.sh /path/to/classification_masks.mat
set -e
MAT=${1:?usage: cf_weizmann_folds.sh classification_masks.mat}
SUBJECTS=(daria denis eli ido ira lena lyova moshe shahar)
MODELS=${MODELS:-"lstm melstm"}
PYTHON=${PYTHON:-python}
for i in "${!SUBJECTS[@]}"; do
  TEST=${SUBJECTS[$i]}
  VAL=${SUBJECTS[$(( (i + 1) % ${#SUBJECTS[@]} ))]}
  for M in $MODELS; do
    case $M in
      lstm)   EXTRA=(--model lstm) ;;
      felstm) EXTRA=(--model felstm --v_range 2) ;;
      melstm) EXTRA=(--model melstm --num_vel_modes 4 --velocity_source frame_pair) ;;
    esac
    NAME="cfw_${M}_test-${TEST}"
    sbatch --job-name="$NAME" --partition=gpu --gres=gpu:1 --time=12:00:00 --mem=32G \
      --cpus-per-task=8 --output="slurm_%x_%j.log" --wrap \
      "$PYTHON -m motion_benchmarks.train_cf_video --dataset weizmann --weizmann_mat $MAT \
        --test_subjects $TEST --val_subjects $VAL ${EXTRA[*]} --readout_steps 6 \
        --forget_bias 0.44 --forget_bias_long 2.97 --control shuffle --run_name $NAME"
  done
done
