#!/bin/sh

DIR="./datasets/datasetB/final_batches/"
mkdir -p $DIR
DIR="./datasets/datasetB/final_batch/"
mkdir -p $DIR

TRAIN="./datasets/datasetB/edges_train_B.csv"
VAL="./datasets/datasetB/input_B_initial.csv"
TEST="./datasets/datasetB/final_input_B.csv"


python3 preprocess.py $TRAIN $VAL $TEST && python3 extra-preprocess.py