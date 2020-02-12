import pandas as pd
from dynamobi import sample_graph, negative_edge_sampling, create_train_graph, filter_test, test_multiple_features
from utils import create_train_test_split


def read_file():
    df = pd.read_csv('data/soc-youtube-growth.edges', comment='%', sep=' ', names=['Source','Target','X','Date'])
    df['Class'] = 1
    df['Source'] = df['Source'].astype(str)
    df['Target'] = df['Target'].astype(str)
    df = df[['Source','Target','Date','Class']]
    df['Date'] = pd.to_datetime(df.Date, unit='s')
    df = df.sort_values('Date')
    df_train = df[(df.Date.dt.year == 2006) | (df.Date.dt.month < 4)]
    df_test = df[(df.Date.dt.year == 2007) & (df.Date.dt.month >= 4)]
    return df_train, df_test

if __name__ == "__main__":
    df_train, df_test = read_file()
    sample_sizes = [100000 + 100000*i for i in range(20)]
    for size in sample_sizes:
        df_train = sample_graph(df_train, size, 'random', g=None)
        g, df_train, df_test = filter_test(df_train, df_test, wcc=False)
        df_train, df_test = negative_edge_sampling(g, df_train, df_test)
        test_multiple_features(
            g,
            df_train,
            df_test,
            paths={},
            print_results=True,
            heuristic=True,
            node2vec=False,
            deepwalk=True)