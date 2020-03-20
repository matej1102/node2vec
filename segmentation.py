

from definitionsV4 import *
from initialization import *
from accuracy import *
import os

from node2vec import Node2Vec


last_iteration = 0
logged_iterations = []
losses_tr = []
corrects_tr = []
solveds_tr = []
losses_ge = []
corrects_ge = []
solveds_ge = []
best_test_loss = 9999
# @title Run training  { form-width: "30%" }

# You can interrupt this cell's training loop at any time, and visualize the
# intermediate results by running the next cell (below). You can then resume
# training by simply executing this cell again.


start_time = time.time()
last_log_time = start_time
input_graphs, target_graphs = generate_networkx_graphs(rand, batch_size_tr, num_nodes_min_max_tr, theta, True)
node2vecs = []
models = []
for index in range(len(input_graphs)):
    node2vecs.append(Node2Vec(input_graphs[index], dimensions=64, walk_length=30, num_walks=200, workers=4))
    models.append(node2vecs[index].fit(window=10, min_count=1, batch_words=4))
    models[index].wv.most_similar('2')
    models[index].wv.save_word2vec_format(EMBEDDING_FILENAME = "./results/"+index++"embeddings.emb")