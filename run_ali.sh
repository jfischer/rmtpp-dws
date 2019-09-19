#!/bin/bash
if [[ "$#" == "1" ]]; then
  EPOCHS=$1
else
  EPOCHS=100
fi
export KMP_DUPLICATE_LIB_OK=TRUE
python src/run_with_lineage.py data/real/ali/event-train.txt data/real/ali/time-train.txt data/real/ali/event-test.txt data/real/ali/time-test.txt --epochs=$EPOCHS --train-eval --test-eval --summary ./tfevents --metrics ./results  --batch-size 10 --restart --cpu-only
