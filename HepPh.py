import pandas as pd
from dynamobi import sample_graph, negative_edge_sampling, create_train_graph, filter_test, test_multiple_features
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

if __name__ == "__main__":
    df_edges = read_file()
    splits = [0.25, 0.5, 0.75]
    for split in splits:
        df_train = df_edges[:int(len(df_edges)*split)]
        df_test = df_edges[int(len(df_edges)*split):]
        g, df_train, df_test = filter_test(df_train, df_test, wcc=True)
        df_train, df_test = negative_edge_sampling(g, df_train, df_test)
        test_multiple_features(
            g,
            df_train,
            df_test,
            paths={},
            print_result=False,
            heuristic=True,
            node2vec=True,
            deepwalk=True)
