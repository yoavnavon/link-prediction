import pandas as pd
from dynamobi import *
from utils import create_train_test_split, test_model
from features import apply_heuristic, train_node2vec, train_deepwalk
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

def read_file():
    edges = pd.read_csv('data/HepPh/cit-HepPh.txt',comment='#',sep='\t', names=['Source','Target'])
    dates = pd.read_csv('data/HepPh/cit-HepPh-dates.txt',comment='#',sep='\t', names=['Source','Date'],dtype={'Source':str})
    # dates['Source'] = dates.Source.apply(lambda i: i[2:] if len(i) > 7 else i)
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
    df_edges = shuffle(df_edges)
    splits = [0.1 * i for i in range(1,10)]
    # print(df_edges)
    for split in splits:
        print(split)
        df_train = df_edges.iloc[:int(len(df_edges)*split)]
        df_test = df_edges.iloc[int(len(df_edges)*split):]
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
    #         'heuristic': 'results/hepph/20_time_heuristic.csv',
    #         'node2vec': 'results/hepph/20_time_node2vec.csv',
    #         'deepwalk': 'results/hepph/20_time_deepwalk.csv'
    #         },
    #         print_results=True,
    #         heuristic=True,
    #         node2vec=True,
    #         deepwalk=True,
    #         resume=False)

    df = read_file()
    df = shuffle(df)
    results_path = 'results/hepph/26_prediction-stats.csv'
    with open(results_path,'w') as file:
        file.write('Size,Method,Split,Common,Hub,%Common\n')
    for split in [0.2 * i for i in range(1,5)]:
        size = int(len(df)*split)
        df_train_full = df.iloc[:size]
        df_test_full = df.iloc[size:]
        prediction_stats(size, df_train_full, df_test_full,results_path)

