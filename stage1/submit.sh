#!/bin/bash

spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 8 \
    code.py \
