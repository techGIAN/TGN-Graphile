# TGN-Graphile
Using <a href="https://github.com/twitter-research/tgn">Rossi et. al.'s</a> TGN Model to perform temporal link prediction task; competition under WSDM 2022 (Graphile is our team name)

## Abstract
[to be filled soon...]

## Hyperparameters
[...]

## Usage
Here are the steps to run this.

### A. Preprocessing the data for TGN's input
1) Create a folder called ```datasets``` in this root directory. Then create two subdirectories under this directory called, ```datasetA``` and ```datasetB```.
2) Download ```edges_train_A.csv``` and ```edges_train_B.csv``` <a href="https://www.dgl.ai/WSDM2022-Challenge/">here</a> and place them in the respective subdirectories. Do the same for ```final_input_A.csv``` and ```final_input_B.csv``` (note that you may need to rename the downloaded files if necessary); the ```initial_input_A.csv``` and ```initial_input_B.csv``` as well.
3) Modify lines 8-10 of ```preprocess.sh``` according to the dataset you would like to run.
4) Run these two lines of code:
```
chmod u+x ./preprocess.sh
./preprocess.sh
```
What you essentially get are batches of datasets coming from the training and testing sets of the original downloaded files, to which we feed to TGN for each batch -- and then concatenate the predictions afterwards.
