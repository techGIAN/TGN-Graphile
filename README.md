# TGN-Graphile
Using <a href="https://github.com/twitter-research/tgn">Rossi et. al.'s</a> TGN Model to perform temporal link prediction task; competition under WSDM 2022 (Graphile is our team name)

## Abstract
<i>[to be filled soon...]</i>

## Hyperparameters
<i>[to be filled soon...]</i>

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

### B. Running TGN
1) Open ```wsdm.sh``` and check lines 17-37 whether changes have to be made (depending on the dataset). Line 17 is the number of batche created from A. Line 20 is the dataset and lines 23-37 are the hyperparameters.
2) Then run using these commands:
```
chmod u+x ./wsdm.sh
./wsdm.sh
```
The predicted probabiltiies for the test data are under ```prob/pos``` whose prefix is ```pos_probX.txt``` and their IDs with ```pos_idX.txt```. We will make necessary postprocessing of the data later for final submission.

<b>Note:</b>
There is a bug that we cannot fix for some reason. So please check. Go to ```prob/pos/pos_idX.txt``` where ```X``` is the last batch. Ensure that the first ID in that file is the first ID of the test data of the last batch (which can be conveniently found under ```data/final_batch/final_BX.csv```. If it is not, then change it. It is important that the dimensions of ```prob/pos/pos_idX.txt``` and ```prob/pos/pos_probX.txt``` are the same. Otherwise, Part C below will fail.
<i>We have resolved this, but still check for any bugs that occur.</i>

### C. Postprocessing
Simply run ```python3 postprocess.py``` and this will output two files ```output_X.csv``` and ```output_verify_X.csv``` under ```prob/``` where ```X``` is etiher ```A``` or ```B```. The first file is the one we submit; the second file is the one we use to check things.

<i>Please contact us for any questions or bugs found. Thank you.</i> 

