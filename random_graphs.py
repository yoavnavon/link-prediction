import networkx as nx
from metrics import get_metrics
import json
from utils import create_train_test_split
from dynamobi import negative_edge_sampling,test_multiple_features

def compute_metrics():
    sample_sizes = [i*50000 for i in range(1,11)]
    p = 1e-5
    results = {}
    for size in sample_sizes:
        print(size)
        g = nx.fast_gnp_random_graph(size, p, seed=1, directed=True)
        metrics = get_metrics(g)
        results[size] = metrics
    with open('results/random/random_p1e5_metrics.json','w') as file:
        json.dump(results, file)

def compute_link_prediction(paths={}, print_results=True, heuristic=True, node2vec=True, deepwalk=True):
    sample_sizes = [i*50000 for i in range(1,11)]
    p = 5e-6
    for size in sample_sizes:
        g_full = nx.fast_gnp_random_graph(2*size, p, seed=1, directed=True)
        df = nx.to_pandas_edgelist(g_full, source='Source', target='Target')
        df['Class'] = 1
        df['Date'] = 1
        df['Source'] = df['Source'].astype(str)
        df['Target'] = df['Target'].astype(str)

        g, df_train, df_test = create_train_test_split(df, 0.5)
        df_train, df_test = negative_edge_sampling(g, df_train, df_test)
        test_multiple_features(
            g,
            df_train,
            df_test,
            paths=paths,
            print_results=print_results,
            heuristic=heuristic,
            node2vec=node2vec,
            deepwalk=deepwalk)



if __name__ == "__main__":
    # compute_metrics()

    compute_link_prediction(
        paths={
            'heuristic': 'results/random/16_random_1pe6_heuristic.csv',
            'node2vec': 'results/random/16_random_1pe6_node2vec.csv',
            'deepwalk': 'results/random/16_random_1pe6_deepwalk.csv'
        }
    )
    
    