#!/bin/sh

DIR="./datasets/dataset$1/final_batches/"
mkdir -p $DIR
DIR="./datasets/dataset$1/final_batch/"
mkdir -p $DIR

TRAIN="./datasets/dataset$1/edges_train_$1.csv"
VAL="./datasets/datasetB$1/input_$1_initial.csv"
TEST="./datasets/dataset$1/final_input_$1.csv"


python3 preprocess.py $TRAIN $VAL $TEST && python3 extra-preprocess.py $1