import pandas as pd
from dynamobi import sample_graph, negative_edge_sampling, create_train_graph, filter_test, test_multiple_features
from utils import create_train_test_split


def read_file():
    df = pd.read_csv('data/socfb-konect/socfb-konect.edges', comment='%', sep=' ', names=['Source','Target'])
    df['Class'] = 1
    df['Source'] = df['Source'].astype(str)
    df['Target'] = df['Target'].astype(str)
    return df

if __name__ == "__main__":
    # df_edges = read_file()
    # splits = [0.25, 0.5, 0.75]
    # for split in splits:
    #     df_train = df_edges[:int(len(df_edges)*split)]
    #     df_test = df_edges[int(len(df_edges)*split):]
    #     g, df_train, df_test = filter_test(df_train, df_test, wcc=True)
    #     df_train, df_test = negative_edge_sampling(g, df_train, df_test)
    #     test_multiple_features(
    #         g,
    #         df_train,
    #         df_test,
    #         paths={},
    #         print_result=False,
    #         heuristic=True,
    #         node2vec=True,
    #         deepwalk=True)
    print(df)