from definitionsV4 import *
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


input_graphs, target_graphs  = generate_networkx_graphs(rand, batch_size_tr, num_nodes_min_max_tr, theta, True)
node2vecs = []
models = []
for index in range(len(input_graphs)):
    node2vecs.append(Node2Vec(input_graphs[index], dimensions=128, walk_length=50, num_walks=20, workers=4))
    models.append(node2vecs[index].fit(window=100, min_count=1, batch_words=4))
    models[index].wv.save_word2vec_format("./results/"+str(index)+"embeddings.emb")