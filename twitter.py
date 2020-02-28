import pandas as pd
from dynamobi import *
from utils import create_train_test_split, test_model
from features import apply_heuristic, train_node2vec, train_deepwalk
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

def read_file():
    df = pd.read_csv('data/twitter_combined.txt', sep=' ', names=['Source','Target'])
    df['Class'] = 1
    df['Source'] = df['Source'].astype(str)
    df['Target'] = df['Target'].astype(str)
    df['Date'] = 1
    df = df.drop_duplicates(subset=['Source','Target'])
    df = shuffle(df)
    return df

def train_test(paths={}, resume=True, print_results=True, heuristic=True, node2vec=True, deepwalk=True):
    # Create Result Files
    if paths and not resume:
        print('Creating Results Files')
        list(map(create_file,paths.values()))

    df_edges = read_file()
    df_edges = shuffle(df_edges)
    sample_sizes = [50000 * i for i in range(1,40)]
    # print(df_edges)
    for size in sample_sizes:
        print(size)
        df_train = df_edges.iloc[:size]
        df_test = df_edges.iloc[size:size+500000]
        g, df_train, df_test = filter_test(df_train, df_test, wcc=False)
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
    # train_test(
    #         paths={
    #         'heuristic': 'results/twitter/21_random_heuristic.csv',
    #         'node2vec': 'results/twitter/21_random_node2vec.csv',
    #         'deepwalk': 'results/twitter/21_random_deepwalk.csv'
    #         },
    #         print_results=True,
    #         heuristic=True,
    #         node2vec=True,
    #         deepwalk=True,
    #         resume=False)

    df = read_file()
    df = shuffle(df)
    results_path = 'results/twitter/26_prediction-stats.csv'
    with open(results_path,'w') as file:
        file.write('Size,Method,Split,Common,Hub,%Common\n')
    for size in [100000 * i for i in range(1,50)]:
        df_train_full = df.iloc[:size]
        df_test_full = df.iloc[size:size+500000]
        prediction_stats(size, df_train_full, df_test_full,results_path)