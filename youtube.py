import pandas as pd
from dynamobi import sample_graph, negative_edge_sampling, create_train_graph, filter_test, test_multiple_features, create_file
from utils import create_train_test_split


def read_file():
    df = pd.read_csv('data/soc-youtube-growth.edges', comment='%', sep=' ', names=['Source','Target','X','Date'])
    df['Class'] = 1
    df['Source'] = df['Source'].astype(str)
    df['Target'] = df['Target'].astype(str)
    df = df[['Source','Target','Date','Class']]
    df['Date'] = pd.to_datetime(df.Date, unit='s')
    df = df.sort_values('Date')
    df = df.drop_duplicates(subset=(['Source','Target']))
    return df

def train_test(paths={}, resume=True, print_results=True, heuristic=True, node2vec=True, deepwalk=True):
    # Create Result Files
    if paths and not resume:
        print('Creating Results Files')
        list(map(create_file,paths.values()))

    df = read_file()
    # print(df.groupby([df['Date'].dt.year, df['Date'].dt.month]).count())

    months = [i for i in range(5)]
    m = 5
    sample_sizes = [1000000 + 1000000*i for i in range(5)]
    df_train_full = df[(df.Date.dt.year == 2006) | (df.Date.dt.month <= m)]
    df_test_full = df[(df.Date.dt.year == 2007) & (df.Date.dt.month > m)]
    print(len(df_train_full),len(df_test_full))
    for size in sample_sizes:
    # for m in months:
        df_test = df_test_full
        df_train = sample_graph(df_train_full, size, 'random')
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
    train_test(
            paths={
            'heuristic': 'results/youtube/14_1M_heuristic.csv',
            'node2vec': 'results/youtube/14_1M_node2vec.csv',
            'deepwalk': 'results/youtube/14_1M_deepwalk.csv'
            },
            print_results=True,
            heuristic=True,
            node2vec=True,
            deepwalk=True,
            resume=False)
    