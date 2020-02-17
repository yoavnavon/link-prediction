import numpy as np
import pandas as pd
import networkx as nx
from dynamobi import sample_graph
from youtube import read_file
import json

def get_metrics(g):
    out_degree = [v for i,v in g.out_degree]
    in_degree = [v for i,v in g.in_degree]
    betweeness =list(nx.betweenness_centrality(g,k=10).values())
    closeness = list(nx.closeness_centrality(g).values())
    clustering = list(nx.clustering(g).values())
    n_degree = [v for v in nx.average_neighbor_degree(g).values()]
    scc = [len(s) for s in nx.strongly_connected_components(g)]
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
        
        'min-betweeness':min(betweeness),
        'max-betweeness':max(betweeness),
        'avg-betweeness':np.mean(betweeness),
        'std-betweeness':np.std(betweeness),
        
        'min-closeness':min(closeness),
        'max-closeness':max(closeness),
        'avg-closeness':np.mean(closeness),
        'std-closeness':np.std(closeness),
        
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

if __name__ == "__main__":
    df = read_file()
    m = 3
    df_full = df[(df.Date.dt.year == 2006) | (df.Date.dt.month <= m)]

    sample_sizes = [i*50000 for i in range(1,11)]
    metrics_results = {}
    for size in sample_sizes:
        print(size)
        df_sample = sample_graph(df_full,size,'random')
        g = nx.from_pandas_edgelist(df_sample, source='Source', target='Target', create_using=nx.DiGraph()) 
        metrics = get_metrics(g)
        metrics_results[size] = metrics
    with open('results/youtube/metrics.json','w') as file:
        json.dump(metrics_results, file)
