from os import listdir
from os.path import isfile, join
import pandas as pd
import sys

X = sys.argv[1]

original_path = './data/final_batch/'
path =  './prob/pos/'
out_path = './prob/'

my_files = [f for f in listdir(original_path) if isfile(join(original_path, f))]
my_files = [x for x in my_files if '._' not in x]
N_batches = len(my_files)
batch_size = 50000
first_test_id = int(batch_size*0.8)

all_df = pd.DataFrame(columns=['id', 'pred'])
for i in range(1, N_batches+1):


    # get the first id of the test data for each batch -- to resolve the bug
    final_batch_data = pd.read_csv(original_path + 'final_{}{}.csv'.format(X, i))
    first_id = int(final_batch_data.iloc[first_test_id,0:1])

    with open(path + 'pos_prob{}.txt'.format(i)) as f:
        lines = f.readlines()

    probs = []
    for line in lines:
        l = line.strip()
        l = l.replace('tensor(', '').replace(')', '')
        probs.append(l)

    with open(path + 'pos_id{}.txt'.format(i)) as f:
        lines = f.readlines()

    iDs = [first_id]
    for line in lines:
        l = line.strip()
        iDs.append(int(l))
    
    
    pred_df = pd.DataFrame({'id': iDs, 'pred': probs}) # contains the probabilities
    pred_df = pred_df.sort_values(by=['id'])



    all_df = pd.concat([all_df, pred_df])

all_df.to_csv(out_path + 'output_verify_{}.csv'.format(X), index=False)  # to verify, not needed
all_df = all_df.drop(['id'], axis=1)
all_df.to_csv(out_path + 'output_{}.csv'.format(X), header=False, index=False)  # for submission