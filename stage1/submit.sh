#!/bin/bash

spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors $1 \
    --executor-cores $2 \
    units.py \
    --dimension $3 \
    --k $4 \
    --totalcore $5
