#!/bin/bash

spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 20 \
    code.py \
