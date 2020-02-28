import pandas as pd
import numpy as np
import networkx as nx
import random
import math
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from glob import glob
import logging  # Setting up the loggings to monitor gensim

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from xgboost import XGBClassifier

from utils import *
from features import *
from sampling import random_walk_sample, node_sampling, bfs_sampling, dfs_sampling

# from pandarallel import pandarallel


def sample_graph(df, size, sampling, g=None):
    """
    Sample nodes form edge list dataframe.
    """
    if sampling == 'time':
        df = df.iloc[0:size]
    if sampling == 'rw':
        g = nx.from_pandas_edgelist(df, source='Source', target='Target', edge_attr='Date', create_using=nx.DiGraph())
        sample = random_walk_sample(g,size)
        df = nx.to_pandas_edgelist(sample, source='Source', target='Target')
    if sampling == 'node':
        df = node_sampling(g, size)
    if sampling == 'bfs':
        df = bfs_sampling(g, size)
    if sampling == 'dfs':
        df = dfs_sampling(g, size)
    if sampling == 'random':
        df = shuffle(df)
        df = df.iloc[0:size]
    df = df.sort_values('Date')
    return df

def read_file(path):
    print(f'Reading file: {path}')
    df = pd.read_csv(path,
        names=['Date','Source','Target','Calls','some','duration','rate'])
    print('Droping Duplicates')
    df = df.drop_duplicates(subset=['Source', 'Target'])
    df['Class'] = 1
    df['Date']= df['Date'].astype('datetime64[ns]')
    df['Source'] = df['Source'].astype(str)
    df['Target'] = df['Target'].astype(str)
    df = df[['Date', 'Source', 'Target','Class']]
    return df

def negative_edge_sampling_train(g, pos_edges, df_train):
    """
    Draw random edges from train dataset
    """
    # Sample negative edges
    nodes = list(g.nodes())
    # source = list(df_train.Source.values[:(len(df_train)//2)]) + random.choices(nodes, k=len(df_train)//2)
    # target = random.choices(nodes, k=len(df_train)//2) + list(df_train.Target.values[:(len(df_train)//2)])
    source = random.choices(nodes, k=len(df_train))
    target = random.choices(nodes, k=len(df_train))
    train_non_edges = set()
    for s,t in zip(source,target):
        if s == t:
            continue
        if (s,t) in train_non_edges: # Repeated random edge
            continue
        if (s,t) in pos_edges: # If random edge is positive
            continue
        train_non_edges.add((s,t))
    
    df_neg_train = pd.DataFrame(list(train_non_edges), columns=['Source', 'Target'])
    df_neg_train['Class'] = 0
    df_train = df_train.append(df_neg_train, sort=True)
    df_train= df_train[['Date','Source', 'Target','Class']]
    return df_train

def negative_edge_sampling_test(g, pos_edges, df_train, df_test):
    """
    Draw random samples from test dataset
    """
    nodes = list(g.nodes())
    # source = list(df_test.Source.values[:(len(df_test)//2)]) + random.choices(nodes, k=len(df_test)//2)
    # target = random.choices(nodes, k=len(df_test)//2) + list(df_test.Target.values[:(len(df_test)//2)])
    source = random.choices(nodes, k=len(df_test))
    target = random.choices(nodes, k=len(df_test))
    test_non_edges = set()
    train_non_edges = set(map(tuple,df_train[['Source','Target']].values))
    for s,t in zip(source,target):
        if s == t:
            continue
        if (s,t) in test_non_edges or (s,t) in train_non_edges: # If random edge already in train or test
            continue
        if (s,t) in pos_edges: # If random edge is positive
            continue
        test_non_edges.add((s,t))
    df_neg_test = pd.DataFrame(list(test_non_edges), columns=['Source', 'Target'])
    df_neg_test['Class'] = 0
    df_test = df_test.append(df_neg_test, sort=True)
    df_test= df_test[['Date','Source', 'Target','Class']]
    return df_test

def negative_edge_sampling(g, df_train, df_test):
    df_combined = pd.concat([df_train,df_test])
    print('pos edges')
    pos_edges = set(map(tuple,df_combined[['Source','Target']].values))
    print('sample negative')
    df_train = negative_edge_sampling_train(g, pos_edges, df_train)
    df_test = negative_edge_sampling_test(g, pos_edges, df_train, df_test)
    print('size train:',len(df_train))
    print('size test:',len(df_test))
    return df_train, df_test

def extract_X_Y(df):
    """
    Extracts features columns from dataframe, assuming the columns are [Source,
    Target, Date, feat1, feat2, ..., featn, Class]
    """
    return (df.iloc[:, 3:len(df.columns)-1], df.iloc[:,len(df.columns)-1])


def test_features(train, test,max_depth=20, gamma=4, scale_pos_weight=1, min_child_weight=1, print_results=True):
    """
    Train and test RF models. XGBoost usually crashes with over 1M edges.
    """
    seed = 2
    # Traditional data
    X_train, Y_train = extract_X_Y(train)
    X_test, Y_test = extract_X_Y(test)

    Y_train= Y_train.astype(int)
    X_train = X_train.astype('float32')
    Y_test= Y_test.astype(int)
    X_test = X_test.astype('float32')

    
    # models 
    # xgb_model=XGBClassifier(
    #     n_estimators=13, 
    #     max_depth=max_depth,
    #     gamma=gamma,
    #     scale_pos_weight=scale_pos_weight,
    #     min_child_weight= min_child_weight,
    #     seed=seed,
    #     # predictor='gpu_predictor',
    #     # tree_method='gpu_hist',
    #     nthread=12)
    rf = RandomForestClassifier(n_estimators= 13, max_depth=max_depth,n_jobs=-1)
    # xgb_results = run_model_test(xgb_model,X_train, Y_train, X_test, Y_test, print_results=print_results) 
    rf_results = run_model_test(rf,X_train, Y_train, X_test, Y_test, print_results=print_results) 
    return len(X_train), rf_results, rf_results

def create_file(filename):
    """
    Create results file with expected headers.
    """
    with open(filename, 'w') as file:
        file.write('size,model,precision,recall,roc_auc,accuracy,f1_score\n')

def save_results(filename,model, size, results):
    """
    Save performance results in csv file
    """
    results = tuple(map(lambda x: round(x,4),results))
    precision, recall, roc_auc, accuracy, f1 = results
    with open(filename,'a') as file:
        file.write(f'{size},{model},{precision},{recall},{roc_auc},{accuracy},{f1}\n')

def print_results(model, size, results):
    results = tuple(map(lambda x: round(x,4),results))
    precision, recall, roc_auc, accuracy, f1 = results
    print('\n')
    print(model)
    print(f'{size} edges')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'ROC AUC: {roc_auc}')
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {accuracy}')




def test_multiple_features(g, df_train, df_test, print_results=False, paths={}, heuristic=True, node2vec=True, deepwalk=True):
    """
    Extract features and test their perfomance.
    """
    
    results = {}
    
    if heuristic:
        print('Heuristic Features')
        df_trad_train, df_trad_test = get_heuristics(g, df_train, df_test)
        size, xgb_results_trad, rf_results_trad = test_features(df_trad_train, df_trad_test, print_results=print_results)
        results['heuristic'] = (size, xgb_results_trad, rf_results_trad)
    
    if node2vec:
        print('Node2Vec Embeddings')
        df_node2vec_train, df_node2vec_test = get_node2vec(g, df_train, df_test, p=1, q=1)
        size, xgb_results_n2v, rf_results_n2v = test_features(df_node2vec_train, df_node2vec_test, print_results=print_results)
        results['node2vec'] = (size, xgb_results_n2v, rf_results_n2v)

    if deepwalk:
        print('Deepwalk Embeddings')
        df_deepwalk_train, df_deepwalk_test = get_deepwalk(g, df_train, df_test)
        size, xgb_results_dw, rf_results_dw = test_features(df_deepwalk_train, df_deepwalk_test, print_results=print_results)
        results['deepwalk'] = (size, xgb_results_dw, rf_results_dw)

    for method,data in results.items():
        size, xgb, rf = data
        if paths:
            save_results(paths[method],'xgb', size//2, xgb)
            save_results(paths[method],'rf', size//2, rf)
    return results

def create_train_graph(df_train, wcc=False):
    """
    Create train graph, and extracs wcc if needed
    """
    g = nx.from_pandas_edgelist(df_train, source='Source', target='Target', create_using=nx.DiGraph()) 
    if wcc:
        wcc = max(nx.weakly_connected_components(g), key=len)
        g = g.subgraph(wcc)
        df_train = df_train[df_train.apply(lambda x: g.has_edge(x['Source'],x['Target']),axis=1)] # reduce df with subgraph

    print(nx.info(g))
    return g, df_train


def filter_test(df_train, df_test, wcc=False): 
    """
    Given a train set, filters test samples useful for training.
    """   
    g, df_train = create_train_graph(df_train, wcc=wcc)

    # Remove unseen nodes from test
    mask = df_test.apply(lambda x: g.has_node(x['Source']) and g.has_node(x['Target']) and not g.has_edge(x['Source'],x['Target']),axis=1)
    df_test = df_test[mask]
    return g, df_train, df_test


def train_test_sampling(train_path, test_path, method='separated', sampling='random', paths={}, print_results=False, sample_sizes=[], resume=False, wcc=False):
    """
    Train and Test features on different data files (Dynamobi). If paths parameter
    is not supplied, it won't write any files.
    """
    
    # Create Result Files
    if paths and not resume:
        print('Creating Results Files')
        list(map(create_file,paths.values()))


    df_train_full = read_file(train_path)
    train_nodes =  set(df_train_full[['Source', 'Target']].values.ravel())
    df_test_full = read_file(test_path)
    mask = df_test_full['Source'].apply(lambda x: x in train_nodes) & df_test_full['Target'].apply(lambda x: x in train_nodes)
    df_test_full = df_test_full[mask]

    df_data = pd.concat([df_train_full,df_test_full])
    G = None
    if sampling in ['node','bfs','dfs']:
        print('Creating Giant Graph')
        G = nx.from_pandas_edgelist(df_data, source='Source', target='Target',edge_attr='Date', create_using=nx.DiGraph())
    
    train_day = df_train_full.Date.dt.day[0]
    test_day = df_test_full.Date.dt.day[0]

    for sample_size in sample_sizes:
        if method == 'combined':
            df_sample = sample_graph(df_data,sample_size,sampling,g=G)
            df_train = df_sample[df_sample.Date.dt.day == train_day]
            df_test = df_sample[df_sample.Date.dt.day == test_day]
            g, df_train, df_test = filter_test(df_train, df_test, wcc=wcc)
        elif method == 'separated':
            df_train = df_train_full.iloc[:sample_size]#sample_graph(df_train_full,sample_size,sampling,g=G)
            df_test = df_train_full.iloc[sample_size:] #df_test_full #sample_graph(df_test_full,sample_size*test_ratio,sampling,g=G)
            g, df_train, df_test = filter_test(df_train, df_test, wcc=wcc)
        

        print('Negative Sampling')
        df_train, df_test = negative_edge_sampling(g, df_train, df_test)
        test_multiple_features(
            g,
            df_train,
            df_test,
            paths=paths,
            print_results=print_results,
            heuristic=True,
            node2vec=True,
            deepwalk=True)
            
def train_model(X, y):
    rf = RandomForestClassifier(n_estimators= 13, max_depth=20,n_jobs=-1)
    rf.fit(X,y)
    return rf


def train_size_full_test(train_size, train_path, results_path):
    """
    Train model in 1 day of sampled dynamobi with given size, and test performance
    on all the other days in the dataset.
    """

    with open(results_path,'w') as file:
        file.write('day,model,method,precision,recall,roc_auc,accuracy,f1_score\n')
    df_full = read_file(train_path)
    df_train = sample_graph(df_full, train_size, 'random')
    g = nx.from_pandas_edgelist(df_train, source='Source', target='Target', create_using=nx.DiGraph()) 
    df_train = negative_edge_sampling_train(g, df_train)
    
    emb_node2vec = train_node2vec(g)
    emb_deepwalk= train_deepwalk(g)

    X_heuristic = apply_heuristic(g, df_train)
    X_node2vec = df_train.apply(lambda x: manual_haddamard(x,emb_node2vec), axis= 1)
    X_deepwalk = df_train.apply(lambda x: manual_haddamard(x,emb_deepwalk), axis= 1)
    Y = df_train['Class']

    rf_heuristic = train_model(X_heuristic,Y)
    rf_node2vec = train_model(X_node2vec,Y)
    rf_deepwalk = train_model(X_deepwalk,Y)

    for test_path in glob('data/dynamobi/*.txt.gz'):
        if test_path == train_path:
            continue
        df_test = read_file(test_path)
        df_test = df_test[df_test.apply(lambda x: g.has_node(x['Source']) and g.has_node(x['Target']) and not g.has_edge(x['Source'],x['Target']),axis=1)]
        df_test = negative_edge_sampling_test(g, df_train, df_test)
        
        results = {}
        X_heuristic_test = apply_heuristic(g, df_test)
        X_node2vec_test = df_test.apply(lambda x: manual_haddamard(x,emb_node2vec), axis= 1)
        X_deepwalk_test = df_test.apply(lambda x: manual_haddamard(x,emb_deepwalk), axis= 1)
        Y_test = df_test['Class'] 

        results['heuristic'] = test_model(rf_heuristic, X_heuristic_test, Y_test)
        results['node2vec'] = test_model(rf_node2vec, X_node2vec_test, Y_test)
        results['deepwalk'] = test_model(rf_deepwalk, X_deepwalk_test, Y_test)
        day = test_path[19:24]
        save_full_test_result(results, day, results_path)

def save_full_test_result(result, day, results_path):
    """
    Save results for train_size_full_test function.
    """
    with open(results_path,'a') as file:
        for method, data in result.items():
            precision, recall, roc_auc, accuracy, f1 = map(lambda x: round(x,4),data)
            file.write(f'{day},rf,{method},{precision},{recall},{roc_auc},{accuracy},{f1}\n')


def train_models(heuristic_train, node2vec_train, deepwalk_train):
    """
    Train RF models with extracted features. 
    """
    heuristic_rf = train_model(*extract_X_Y(heuristic_train))
    node2vec_rf = train_model(*extract_X_Y(node2vec_train))
    deepwalk_rf = train_model(*extract_X_Y(deepwalk_train))
    return heuristic_rf, node2vec_rf, deepwalk_rf




def get_links_stats(g, heuristic_test, node2vec_test, deepwalk_test):
    """
    Compute some edge specific stats for edge prediction quality.
    """

    heuristic_test['Hub'] = heuristic_test.Target.apply(g.in_degree)
    node2vec_test['Hub'] = node2vec_test.Target.apply(g.in_degree)
    deepwalk_test['Hub'] = deepwalk_test.Target.apply(g.in_degree)
    common = lambda x: len(set(g.neighbors(x['Source'])).intersection(set(g.neighbors(x['Target']))))
    heuristic_test['Common'] = heuristic_test.apply(common, axis=1) 
    node2vec_test['Common'] = node2vec_test.apply(common, axis=1) 
    deepwalk_test['Common'] = deepwalk_test.apply(common, axis=1) 

    return heuristic_test, node2vec_test, deepwalk_test

def get_avg_link_stats(*dfs):
    """
    Agregate link stats, returns:
    - Mean number of common neighbors between nodes in edge.
    - Mean in-degree of the target node to detect Hubs.
    - Percentage of edges sharing 0 common neighbors.
    - Number of different target nodes in the prediction.
    """

    for df in dfs:
        yield (
            df['Common'].mean(),
            df['Hub'].mean(),
            100*len(df[df['Common'] == 0])/len(df),
            df['Target'].nunique()
            )

def split_predictions(df):
    """
    Split True Positive, True Negative, False Positive, False Negative, Predicted
    Positives and Predicted Negatives.
    """

    tp = df[(df['Predicted'] == df['Class'])&(df['Predicted']== 1)]
    tn = df[(df['Predicted'] == df['Class'])&(df['Predicted']== 0)]
    fp = df[(df['Predicted'] != df['Class'])&(df['Predicted']== 1)]
    fn = df[(df['Predicted'] != df['Class'])&(df['Predicted']== 0)]
    pos_pred = df[df['Predicted']== 1]
    neg_pred = df[df['Predicted']== 0]
    return tp, tn, fp, fn, pos_pred, neg_pred

def prediction_stats_size(sizes,df_train_full, df_test_full, results_path):
    """
    Compute edge prediction stats for an arange of sample sizes.
    """
    
    with open(results_path,'w') as file:
        file.write('Size,Method,Split,Common,Hub,%Common,NUnique\n')

    for size in sizes:
        prediction_stats(size, df_train_full, df_test_full, results_path)


def prediction_stats(train_size, df_train_full, df_test_full, results_path):
    """
    Compute edge prediction stats for a certain size.
    """

    df_train = sample_graph(df_train_full,train_size,'random')
    df_test = df_test_full #sample_graph(df_test_full,sample_size*test_ratio,sampling,g=G)
    g, df_train, df_test = filter_test(df_train, df_test, wcc=False)
    df_train, df_test = negative_edge_sampling(g, df_train, df_test)

    heuristic_train, heuristic_test = get_heuristics(g, df_train, df_test)
    node2vec_train, node2vec_test = get_node2vec(g, df_train, df_test, p=1, q=1)
    deepwalk_train, deepwalk_test = get_deepwalk(g, df_train, df_test)
    
    heuristic_rf, node2vec_rf, deepwalk_rf = train_models(heuristic_train, node2vec_train, deepwalk_train)
    heuristic_test['Predicted'] = heuristic_rf.predict(extract_X_Y(heuristic_test)[0])
    node2vec_test['Predicted'] = node2vec_rf.predict(extract_X_Y(node2vec_test)[0])
    deepwalk_test['Predicted'] = deepwalk_rf.predict(extract_X_Y(deepwalk_test)[0])
    heuristic_test, node2vec_test, deepwalk_test = get_links_stats(g, heuristic_test, node2vec_test, deepwalk_test)
    
    order = ['TP', 'TN', 'FP', 'FN', 'POS', 'NEG']
    with open(results_path,'a') as file:
        for o,stat in zip(order,get_avg_link_stats(*split_predictions(heuristic_test))):
            common, hub, per_common, nunique =  map(lambda x: round(x,4),stat)
            file.write(f'{train_size},heuristic,{o},{common},{hub},{per_common},{nunique}\n')
        for o,stat in zip(order,get_avg_link_stats(*split_predictions(node2vec_test))):
            common, hub, per_common, nunique  =  map(lambda x: round(x,4),stat)
            file.write(f'{train_size},node2vec,{o},{common},{hub},{per_common},{nunique}\n')
        for o,stat in zip(order,get_avg_link_stats(*split_predictions(deepwalk_test))):
            common, hub, per_common, nunique  =  map(lambda x: round(x,4),stat)
            file.write(f'{train_size},deepwalk,{o},{common},{hub},{per_common},{nunique}\n')

    

if __name__ == "__main__":
    # logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
    
    # sample_sizes = [1000000 + 1000000*i for i in range(10)] #1M
    sample_sizes = [100000*i for i in range(1,50)] #random
    
    # train_test_sampling(
    #     'data/dynamobi/2008-08-01.txt.gz',
    #     'data/dynamobi/2008-08-02.txt.gz',
    #     sample_sizes=sample_sizes,
    #     sampling='random',
    #     wcc=False,
    #     paths={
    #         'heuristic': 'results/dynamobi/24_time_500k_heuristic.csv',
    #         'node2vec': 'results/dynamobi/24_time_500k_node2vec.csv',
    #         'deepwalk': 'results/dynamobi/24_time_500k_deepwalk.csv'
    #     },
    #     print_results=True,
    #     resume=False,
    #     method='separated')

    # train_size_full_test(600000,'data/dynamobi/2008-07-28.txt.gz','results/dynamobi/20_train0728_600k.csv')

    df_train_full = read_file('data/dynamobi/2008-08-01.txt.gz')
    df_test_full = read_file('data/dynamobi/2008-08-02.txt.gz')
    prediction_stats_size(sample_sizes, df_train_full, df_test_full,'results/dynamobi/28_prediction-stats.csv')

    print('DONE')