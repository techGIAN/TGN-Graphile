#!/bin/sh

now=$(date +"%T")
echo "Current time : $now"

DIR="./data/"
mkdir -p $DIR

SRC_DIR="../datasets/datasetB/final_batch/"
cp -r $SRC_DIR $DIR

# MY_FILES=ls ./data/final_batch/ | wc -l
# echo $n_batches
# exit 0

i=2
n_batches=131 # need to specify number of batches + 1
while [ $i -lt $n_batches ]
do
    dataset="final_B$i"

    python3 utils/preprocess_data.py --data $dataset && \
    python3 train_self_supervised.py \
        -d $dataset \
        --n_epoch 1 \
        --use_memory \
        --prefix tgn-attn-final_B \
        --n_runs 1 \
        --n_degree 5 \
        --n_head 8 \
        --n_layer 2 \
        --time_dim 250 \
        --different_new_nodes \
        --backprop_every 2 \
        --message_dim 200 \
        --embedding_module graph_sum \
        --message_function mlp
    
    echo "TGN Batch $i complete"
    i=$(( i+1 ))
done

now=$(date +"%T")
echo "Current time : $now"