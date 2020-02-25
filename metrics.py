import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from dynamobi import sample_graph#, read_file
from youtube import read_file
# from HepPh import read_file
import json
import sys
sys.path.append('./GraphEmbedding')
from ge import Node2Vec, DeepWalk

def get_metrics(g):
    print('degree')
    out_degree = [v for i,v in g.out_degree]
    in_degree = [v for i,v in g.in_degree]
    # print('betweenness')
    # betweeness =list(nx.betweenness_centrality(g,k=10).values())
    # print('closeness')
    # closeness = list(nx.closeness_centrality(g).values())
    print('clustering')
    clustering = list(nx.clustering(g).values())
    print('neighboor degree')
    n_degree = [v for v in nx.average_neighbor_degree(g).values()]
    print('scc')
    scc = [len(s) for s in nx.strongly_connected_components(g)]
    print('wcc')
    wcc = [len(s) for s in nx.weakly_connected_components(g)]
    
    metrics = {
        'min-out-degree':min(out_degree),
        'max-out-degree':max(out_degree),
        'avg-out-degree':np.mean(out_degree),
        'std-out-degree':np.std(out_degree),
        
        'min-in-degree':min(in_degree),
        'max-in-degree':max(in_degree),
        'avg-in-degree':np.mean(in_degree),
        'std-in-degree':np.std(in_degree),
        
        # 'min-betweeness':min(betweeness),
        # 'max-betweeness':max(betweeness),
        # 'avg-betweeness':np.mean(betweeness),
        # 'std-betweeness':np.std(betweeness),

        'min-clustering':min(clustering),
        'max-clustering':max(clustering),
        'avg-clustering':np.mean(clustering),
        'std-clustering':np.std(clustering),
        
        # 'min-closeness':min(closeness),
        # 'max-closeness':max(closeness),
        # 'avg-closeness':np.mean(closeness),
        # 'std-closeness':np.std(closeness),
        
        'min-neighboor-degree':min(n_degree),
        'max-neighboor-degree':max(n_degree),
        'avg-neighboor-degree':np.mean(n_degree),
        'std-neighboor-degree':np.std(n_degree),
        
        'count-scc':len(scc),
        'min-scc':min(scc),
        'max-scc':max(scc),
        'avg-scc':np.mean(scc),
        'std-scc':np.std(scc),
        
        'count-wcc':len(wcc),
        'min-wcc':min(wcc),
        'max-wcc':max(wcc),
        'avg-wcc':np.mean(wcc),
        'std-wcc':np.std(wcc),
    }
    return metrics

def get_walks_stats(walks):
    stats = defaultdict(lambda: 0)
    for walk in walks:
        stats[len(walk)] += 1
    return stats

def get_walks_avg_len(walks):
    return sum(map(len,walks))/len(walks)


if __name__ == "__main__":
    # df_full = read_file('data/dynamobi/2008-08-01.txt.gz')
    df_full = read_file()
    m = 3
    df_full = df_full[(df_full.Date.dt.year == 2006) | (df_full.Date.dt.month <= m)]

    sample_sizes = [i*100000 for i in range(1,10)]
    metrics_results = {}
    for size in sample_sizes:
        print(size)
        df_sample = sample_graph(df_full,size,'random')
        g = nx.from_pandas_edgelist(df_sample, source='Source', target='Target', create_using=nx.Graph()) 
        node2vec = Node2Vec(g, walk_length=80, num_walks=10, p=1, q=1, workers=5)
        deepwalk = DeepWalk(g, walk_length=80, num_walks=10, workers=5)
        node2vec_avg = get_walks_avg_len(node2vec.sentences)
        deepwalk_avg = get_walks_avg_len(deepwalk.sentences)
        print(node2vec_avg, deepwalk_avg)
        metrics_results[size] = (node2vec_avg, deepwalk_avg)
        # metrics = get_metrics(g)
        # metrics_results[size] = metrics
    # with open('results/hepph/20_metrics_walks.json','w') as file:
    #     json.dump(metrics_results, file)


