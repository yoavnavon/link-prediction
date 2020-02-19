import pandas as pd
from dynamobi import sample_graph, negative_edge_sampling, create_train_graph, filter_test, test_multiple_features, create_file
from utils import create_train_test_split


def read_file():
    edges = pd.read_csv('data/HepPh/cit-HepPh.txt',comment='#',sep='\t', names=['Source','Target'])
    dates = pd.read_csv('data/HepPh/cit-HepPh-dates.txt',comment='#',sep='\t', names=['Source','Date'],dtype={'Source':str})
    dates['Source'] = dates.Source.apply(lambda i: i[2:] if len(i) > 7 else i)
    dates['Source'] = dates['Source'].astype(int)
    df = pd.merge(edges, dates, how='inner', on= 'Source')
    df['Date']= df['Date'].astype('datetime64[ns]')
    df['Class'] = 1
    df['Source'] = df['Source'].astype(str)
    df['Target'] = df['Target'].astype(str)
    df = df.sort_values('Date')
    return df

def train_test(paths={}, resume=True, print_results=True, heuristic=True, node2vec=True, deepwalk=True):
    # Create Result Files
    if paths and not resume:
        print('Creating Results Files')
        list(map(create_file,paths.values()))

    df_edges = read_file()
    splits = [0.1 * i for i in range(1,10)]
    for split in splits:
        df_train = df_edges[:int(len(df_edges)*split)]
        df_test = df_edges[int(len(df_edges)*split):]
        print(split)
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
            'heuristic': 'results/hepph/16_time_heuristic.csv',
            'node2vec': 'results/hepph/16_time_node2vec.csv',
            'deepwalk': 'results/hepph/16_time_deepwalk.csv'
            },
            print_results=True,
            heuristic=True,
            node2vec=True,
            deepwalk=True,
            resume=False)
