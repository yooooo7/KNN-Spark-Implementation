#!/bin/bash

spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 6 \
    units.py \
