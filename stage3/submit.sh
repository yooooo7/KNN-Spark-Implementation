#!/bin/bash

spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 4 \
    ulits.py \
