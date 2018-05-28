#!/bin/bash

spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors $3 \
    --executor-cores $4 \
    units.py \
    --dimension $1 \
    --k $2
