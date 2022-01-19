import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from os import listdir
from os.path import isfile, join


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200, test_ids=None):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    TEST_BATCH_SIZE = num_test_instance if test_ids is not None else TEST_BATCH_SIZE # only one batch for test
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    s = ''
    ss = ''
    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      # try:
      if test_ids is not None:
        pos_probs = list(np.squeeze(pos_prob))
        for item in pos_probs:
          ln = str(item)
          s += ln + '\n'
        pos_path = './prob/pos/'
        my_files = [f for f in listdir(pos_path) if isfile(join(pos_path, f))]
        my_files = [x for x in my_files if '._' not in x]
        my_files = [x for x in my_files if 'pos_prob' in x]
        my_files = [int(x.replace('pos_prob','').replace('.txt', '')) for x in my_files if 'pos_prob' in x]
        num = 1 if len(my_files) == 0 else max(my_files)+1

        f = open(pos_path + 'pos_prob' + str(num) + '.txt', 'a')
        f.write(s)
        f.close()
        for ID in test_ids:
          ss += str(ID) + '\n'
        f = open(pos_path + 'pos_id' + str(num) + '.txt', 'a')
        f.write(ss)
        f.close()

      # except:
        # pass

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc
