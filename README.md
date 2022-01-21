# TGN-Graphile
Using <a href="https://github.com/twitter-research/tgn">Rossi et. al.'s</a> TGN Model to perform temporal link prediction task; competition under <a href="https://www.dgl.ai/WSDM2022-Challenge/">WSDM 2022</a>  (Graphile is our team name)

## Abstract
We used the framework TGN developed by Rossi et. al. (2020). Basically it has a module that stores the up-to-date state of each node the model has seen so far, which could help with the temporal link prediction task. After preprocessing the data, we trained the model and did the prediction using the tuned parameters for dataset A and dataset B respectively.

## Package Dependencies
Please use ```python >= 3.7``` along with the following packages installed:
```
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
```


## Hyperparameters
For dataset A, here are the hyperparameters:
```
--n_epoch 1 \
--use_memory \
--prefix tgn-attn-final_A \
--n_runs 1 \
--n_degree 20 \
--n_head 2 \
--n_layer 2 \
--time_dim 200 \
--different_new_nodes \
--backprop_every 2 \
--message_dim 200 \
--embedding_module time \
--aggregator mean
```
For dataset B, here are the hyperparameters:
```
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
```
Please refer <a href="https://github.com/twitter-research/tgn#general-flags">here</a> for a list of the parameters used by TGN and their descriptions. Please refer below to how to use it.

## Usage
Here are the steps to run this.

### A. Preprocessing the data for TGN's input
1) Create a folder called ```datasets``` in this root directory. Then create two subdirectories under this directory called, ```datasetA``` and ```datasetB```.
2) Download ```edges_train_A.csv``` and ```edges_train_B.csv``` <a href="https://www.dgl.ai/WSDM2022-Challenge/">here</a> and place them in the respective subdirectories. Do the same for ```final_input_A.csv``` and ```final_input_B.csv``` (note that you may need to rename the downloaded files if necessary); the ```initial_input_A.csv``` and ```initial_input_B.csv``` as well.
3) Modify lines 8-10 of ```preprocess.sh``` according to the dataset you would like to run.
4) Run these two lines of code:
```
chmod u+x ./preprocess.sh
./preprocess.sh X
```
where ```X``` is either ```A``` or ```B``` for dataset A or B respectively.

What you essentially get are batches of datasets coming from the training and testing sets of the original downloaded files, to which we feed to TGN for each batch -- and then concatenate the predictions afterwards.

### B. Running TGN
1) Open ```wsdm.sh``` and modify lines 26-38 depending on the hyperparameters set above (see Hyperparameter section above).
2) Then run using these commands:
```
chmod u+x ./wsdm.sh
./wsdm.sh X
```
where ```X``` is either ```A``` or ```B``` for dataset A or B respectively.

The predicted probabiltiies for the test data are under ```prob/pos``` whose prefix is ```pos_probX.txt``` and their IDs with ```pos_idX.txt```. We will make necessary postprocessing of the data later for final submission.

<b>Note:</b> <br>
There is a bug that we cannot fix for some reason. So please check. Go to ```prob/pos/pos_idX.txt``` where ```X``` is the last batch. Ensure that the first ID in that file is the first ID of the test data of the last batch (which can be conveniently found under ```data/final_batch/final_BX.csv```). If it is not, then change it. It is important that the dimensions of ```prob/pos/pos_idX.txt``` and ```prob/pos/pos_probX.txt``` are the same. Otherwise, Part C below will fail. <br>
<i>We have resolved this, but still check for any bugs that occur.</i>

### C. Postprocessing
Simply run ```python3 postprocess.py X``` (where ```X``` is either ```A``` or ```B``` for dataset A or B respectively) and this will output two files ```output_X.csv``` and ```output_verify_X.csv``` under ```prob/```. The former file is the one we submit; the latter file is the one we use to check things but is not necessary for submission.

<i>Please contact us for any questions or bugs found. Thank you.</i> 

