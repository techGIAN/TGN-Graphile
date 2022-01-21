import pandas as pd
import random as rnd
import sys

def one_hot(edge_type, unique_edge_types):
    return ','.join([str(int(x == edge_type)) for x in unique_edge_types])

def time_aggregate(start_time, end_time, method):
    if method == 'start':
        return start_time
    elif method == 'end':
        return end_time
    elif method == 'mean':
        return int(start_time/2 + end_time/2)
    elif method == 'random':
        return [start_time, end_time][rnd.randint(0,1)]
    elif method == 'random_between':
        return rnd.randint(start_time, end_time)
    else:
        return None

def corrupt(df, src, dst, edge_type):
    min_timestamp = df['timestamp'].min()
    max_timestamp = df['timestamp'].max()

    positive_values = df[(df['user_id'] == src) & (df['item_id'] == dst) & (df['edge_type'] == edge_type)]['timestamp'].values

    chosen_time = positive_values[0]
    while chosen_time in positive_values:   # keep choosing another time within the range until a negative timestamp is chosen
        chosen_time = rnd.randint(min_timestamp, max_timestamp)
    return chosen_time

def df_one_hot(x, edge_types):
    return one_hot(x, edge_types)

def df_corrupt(x, temp_df):
    min_timestamp = temp_df['timestamp'].min()
    max_timestamp = temp_df['timestamp'].max()
    user, item, edge_type = x['user_id'], x['item_id'], x['edge_type']

    positive_values = temp_df[(temp_df['user_id'] == user) & (temp_df['item_id'] == item) & (temp_df['edge_type'] == edge_type)]['timestamp'].values

    chosen_time = positive_values[0]
    while chosen_time in positive_values:   # keep choosing another time within the range until a negative timestamp is chosen
        chosen_time = rnd.randint(min_timestamp, max_timestamp)
    return chosen_time

def df_time_aggregate(x, method):
    return time_aggregate(x['start_time'], x['end_time'], method)

arguments = sys.argv
train_path = arguments[1]
initial_path = arguments[2]
test_path = arguments[3]
data_type = test_path[-5]

# GET DISTRIBUTION OF HOW MUCH 1s ans 0s ARE NEEDED
df = pd.read_csv(initial_path, header=None)
df.columns = ['user_id', 'item_id', 'edge_type',
                'start_time', 'end_time', 'state_label']

ones = df[df['state_label'] == 1].shape[0]
zeros = df[df['state_label'] == 0].shape[0]
total = df.shape[0] #3863
batch_size = total*2 #7726
batch_size = 50000 # We incresed to 50,000 per batch
test_sample_amount = int(batch_size*0.2) # 1545
N_batches = int(200000/test_sample_amount) #129

ones_percent = ones/total
zeros_percent = zeros/total

count = 0
# TRAINING 
train_df = pd.read_csv(train_path, header=None)
train_df.columns = ['user_id', 'item_id', 'edge_type', 'timestamp', 'feat']
train_df = train_df.drop(['feat'], axis=1)
train_copy_df = train_df.copy(deep=True)

# TESTING
test_df = pd.read_csv(test_path, header=None)
test_df.columns = ['user_id', 'item_id', 'edge_type',
                'start_time', 'end_time']
test_copy_df = test_df.copy(deep=True)
earliest_timestamp = test_df['start_time'].min()
edge_types = test_df['edge_type'].unique()
edge_types.sort()

train_df = train_df[train_df['timestamp'] < earliest_timestamp]

for i in range(1, N_batches+2):
    
    test_sample_amount = 200000 - test_sample_amount*N_batches if i == N_batches+1 else test_sample_amount # if = 695, else = 1545
    train_sample_amount = int(test_sample_amount/0.2*0.8) if i == N_batches+1 else batch_size-test_sample_amount # if = 2780, else = 6181
    train_ones_batch = int(ones_percent*train_sample_amount) # if = 1619, else = 3601
    train_zeros_batch = train_sample_amount-train_ones_batch # if = 1161, else = 2580

    if test_sample_amount == 0:
        break # extra remainder batch not needed because remainder is 0.

    # TRAINING
    indices = list(train_df.index)
    positive_ix = rnd.sample(indices, k=train_ones_batch)
    indices = list(set(indices).difference(set(positive_ix)))
    negative_ix = rnd.sample(indices, k=train_zeros_batch)

    pos_df = train_df.iloc[positive_ix].reset_index().drop(['index'], axis=1)
    neg_df = train_df.iloc[negative_ix].reset_index().drop(['index'], axis=1)
    pos_df['state_label'] = 1
    neg_df['state_label'] = 0

    pos_df['comma_separated_list_of_features'] = pos_df['edge_type'].apply(df_one_hot, args=(edge_types,))
    neg_df['comma_separated_list_of_features'] = neg_df['edge_type'].apply(df_one_hot, args=(edge_types,))

    temp_neg = neg_df.copy(deep=True)
    neg_df['timestamp'] = neg_df.apply(df_corrupt, args=(temp_neg,), axis=1)

    new_df = pd.concat([pos_df, neg_df])
    new_df = new_df.reset_index().drop(['index'], axis=1)
    new_df = new_df.drop(['edge_type'], axis=1)


    # TESTING
    low_ix = test_sample_amount*(i-1)
    high_ix = test_sample_amount*i

    df = test_df.copy(deep=True)
    df = df.iloc[low_ix:high_ix,:]
    df['timestamp'] = 0
    df['timestamp'] = df.apply(df_time_aggregate, args=('random_between',), axis=1)
    df['state_label'] = 1
    df['comma_separated_list_of_features'] = df['edge_type'].apply(df_one_hot, args=(edge_types,))
    df = df.drop(['edge_type', 'start_time', 'end_time'], axis=1)


    new_df = pd.concat([new_df, df])
    new_df = new_df.reset_index().drop(['index'], axis=1)
    new_df = new_df.reset_index().rename(columns={'index':'id'})
    new_df['id'] = new_df['id'] + count
    count = new_df.shape[0]*i

    new_df = new_df.sort_values(by=['timestamp'])
    new_df = new_df.reset_index().drop(['index'], axis=1)



    new_df['id'] = new_df['id'].astype('int64')
    new_df['timestamp'] = new_df['timestamp'].astype('int64')
    new_df['user_id'] = new_df['user_id'].astype('int64')
    new_df['item_id'] = new_df['item_id'].astype('int64')
    new_df['state_label'] = new_df['state_label'].astype('int64')
    new_df['comma_separated_list_of_features'] = new_df['comma_separated_list_of_features'].astype('str')

    new_df.to_csv('./datasets/dataset' + data_type + '/final_batches/final_' + data_type + str(i) + '.csv', index=False)
    print('Batch {} complete'.format(i))


    