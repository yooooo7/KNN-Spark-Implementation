#!/bin/bash

spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 10 \
    ulits.py \
