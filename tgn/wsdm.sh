#!/bin/sh

start=$(date +"%T")
echo "Start time : $start"

DIR="./data/final_batch"
mkdir -p $DIR

SRC_DIR="../../datasets/dataset$1/final_batch/"
cp -r $SRC_DIR $DIR

ITEMS=$(ls $DIR | wc -l | tr -s " ")
ITEMS=${ITEMS%% }
ITEMS=${ITEMS## }

i=1
n_batches="$ITEMS"

while [ $i -le $n_batches ]
do
    dataset="final_$1$i"

    python3 utils/preprocess_data.py --data $dataset && \
    python3 train_self_supervised.py \
        -d $dataset \
        --n_epoch 1 \
        --use_memory \
        --prefix tgn-attn-final_$1 \
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

end=$(date +"%T")
echo "End time : $end"

runtime=$((end-start))
echo "Total runtime: $runtime"