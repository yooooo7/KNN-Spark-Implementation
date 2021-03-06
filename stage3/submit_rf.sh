#!/bin/bash

spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 4 \
    --executor-cores 1 \
    rf.py \
    --treenums $1
