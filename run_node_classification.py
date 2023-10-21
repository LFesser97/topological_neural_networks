from attrdict import AttrDict
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid, HeterophilousGraphDataset
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, dropout_edge
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected
from experiments.node_classification import Experiment

import time
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, borf

import torch_geometric.transforms as T
from torch_geometric.transforms import Compose
from custom_encodings import ShortestPathGenerator, OneHotEdgeAttr, LocalCurvatureProfile


# positional and structural encodings
# transform = T.AddRandomWalkPE(walk_length=16)
# print("Encoding Random Walk PE")

transform = T.AddLaplacianEigenvectorPE(k=8)
print("Encoding Laplacian Eigenvector PE")

# transform = T.RootedRWSubgraph(walk_length=10)
# print("Encoding Rooted RW Subgraph")

# transform = T.Compose([T.RootedRWSubgraph(walk_length=10), T.AddRandomWalkPE(walk_length=16)])
# print("Encoding Rooted RW Subgraph + Random Walk PE")

# transform = T.Compose([T.RootedRWSubgraph(walk_length=10), T.AddLaplacianEigenvectorPE(k=8)])
# print("Encoding Rooted RW Subgraph + Laplacian Eigenvector PE")

# lcp = LocalCurvatureProfile()
# print(f"Encoding Local Curvature Profile (ORC)")

# transform = lcp.compute_orc_approx


largest_cc = LargestConnectedComponents()
# cornell = WebKB(root="data", name="Cornell", transform=transform)
# wisconsin = WebKB(root="data", name="Wisconsin", transform=transform)
# texas = WebKB(root="data", name="Texas", transform=transform)
# chameleon = WikipediaNetwork(root="data", name="chameleon")
# cora = Planetoid(root="data", name="cora")
# citeseer = Planetoid(root="data", name="citeseer")
# roman_empire = HeterophilousGraphDataset(root="data", name="Roman-empire")
amazon_ratings = HeterophilousGraphDataset(root="data", name="Amazon-ratings", transform=transform)
minesweeper = HeterophilousGraphDataset(root="data", name="Minesweeper", transform=transform)
tolokers = HeterophilousGraphDataset(root="data", name="Tolokers", transform=transform)
# questions = HeterophilousGraphDataset(root="data", name="Questions")

# datasets = {"cora": cora, "citeseer": citeseer, "chameleon": chameleon}

# datasets = {"cornell": cornell, "wisconsin": wisconsin, "texas": texas, 
            #"chameleon": chameleon, "cora": cora, "citeseer": citeseer}

datasets = {"amazon_ratings": amazon_ratings, "minesweeper": minesweeper, "tolokers": tolokers}

for key in datasets:
    dataset = datasets[key]
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)

def log_to_file(message, filename="results/node_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

default_args = AttrDict({
    "dropout": 0.2,
    "num_layers": 8,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "layer_type": "R-GCN",
    "display": True,
    "num_trials": 10,
    "eval_every": 1,
    "rewiring": "fosr",
    "num_iterations": 50,
    "num_relations": 2,
    "patience": 100,
    "dataset": None,
    "borf_batch_add" : 4,
    "borf_batch_remove" : 2,
    "sdrf_remove_edges" : False
})


results = []
args = default_args
args += get_args_from_input()

if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}

for key in datasets:
    accuracies = []
    print(f"TESTING: {key} ({args.rewiring})")
    dataset = datasets[key]

    
    start = time.time()
    if args.rewiring == "fosr":
        edge_index, edge_type, _ = fosr.edge_rewire(dataset.data.edge_index.numpy(), num_iterations=args.num_iterations)
        dataset.data.edge_index = torch.tensor(edge_index)
        dataset.data.edge_type = torch.tensor(edge_type)
        print(dataset.data.num_edges)
        print(len(dataset.data.edge_type))
    elif args.rewiring == "sdrf_bfc":
        curvature_type = "bfc"
        dataset.data.edge_index, dataset.data.edge_type = sdrf.sdrf(dataset.data, loops=args.num_iterations, remove_edges=args.sdrf_remove_edges, 
                is_undirected=True, curvature=curvature_type)
    elif args.rewiring == "borf":
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
        print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.borf_batch_remove}")
        dataset.data.edge_index, dataset.data.edge_type = borf.borf3(dataset.data, 
                loops=args.num_iterations, 
                remove_edges=False, 
                is_undirected=True,
                batch_add=args.borf_batch_add,
                batch_remove=args.borf_batch_remove,
                dataset_name=key,
                graph_index=0)
    elif args.rewiring == "barf_3":
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
        print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
        print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
        for i in range(len(dataset)):
            dataset[i].edge_index, dataset[i].edge_type = borf.borf4(dataset[i], 
                    loops=args.num_iterations, 
                    remove_edges=False, 
                    is_undirected=True,
                    batch_add=args.borf_batch_add,
                    batch_remove=args.borf_batch_remove,
                    dataset_name=key,
                    graph_index=i)
    elif args.rewiring == "barf_4":
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
        print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
        print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
        for i in range(len(dataset)):
            dataset[i].edge_index, dataset[i].edge_type = borf.borf5(dataset[i], 
                    loops=args.num_iterations, 
                    remove_edges=False, 
                    is_undirected=True,
                    batch_add=args.borf_batch_add,
                    batch_remove=args.borf_batch_remove,
                    dataset_name=key,
                    graph_index=i)
    elif args.rewiring == "sdrf_orc":
        curvature_type = "orc"
        dataset.data.edge_index, dataset.data.edge_type = sdrf.sdrf(dataset.data, loops=args.num_iterations, remove_edges=False, 
                is_undirected=True, curvature=curvature_type)
        
    elif args.rewiring == "dropedge":
        p = 0.8
        print(f"[INFO] Dropping edges with probability {p}")
        for i in range(len(dataset)):
            dataset[i].edge_index, dataset[i].edge_type = dropout_edge(dataset[i].edge_index, dataset[i].edge_type, p=p, force_undirected=True)

    end = time.time()
    rewiring_duration = end - start
    print(f"Rewiring duration: {rewiring_duration}")
    
    """
    # print(rewiring.spectral_gap(to_networkx(dataset.data, to_undirected=True)))
    start = time.time()
    for trial in range(args.num_trials):
        print(f"TRIAL #{trial+1}")
        test_accs = []
        for i in range(args.num_splits):
            train_acc, validation_acc, test_acc = Experiment(args=args, dataset=dataset).run()
            test_accs.append(test_acc)
        test_acc = max(test_accs)
        accuracies.append(test_acc)
    end = time.time()
    run_duration = end - start

    log_to_file(f"RESULTS FOR {key} ({args.rewiring}):\n")
    log_to_file(f"average acc: {np.mean(accuracies)}\n")
    log_to_file(f"plus/minus:  {2 * np.std(accuracies)/(args.num_trials ** 0.5)}\n\n")
    results.append({
        "dataset": key,
        "rewiring": args.rewiring,
        "num_iterations": args.num_iterations,
        "borf_batch_add" : args.borf_batch_add,
        "borf_batch_remove" : args.borf_batch_remove,
        "avg_accuracy": np.mean(accuracies),
        "ci":  2 * np.std(accuracies)/(args.num_trials ** 0.5),
        "run_duration" : run_duration,
        #"rewiring_duration" : rewiring_duration
    })
    results_df = pd.DataFrame(results)
    with open(f'results/node_classification_{args.layer_type}_{args.rewiring}.csv', 'a') as f:
        results_df.to_csv(f, mode='a', header=f.tell()==0)
    """