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
    if sampling == 'time':
        df = df.iloc[0:size]
    if sampling == 'rw':
        print('creating graph')
        g = nx.from_pandas_edgelist(df, source='Source', target='Target', edge_attr='Date', create_using=nx.DiGraph())
        print('sampling')
        sample = random_walk_sample(g,size)
        print('creating df')
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

def negative_edge_sampling_train(g, df_train):
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
        if (s,t) in train_non_edges:
            continue
        if g.has_edge(s,t):
            continue
        train_non_edges.add((s,t))
    
    df_neg_train = pd.DataFrame(list(train_non_edges), columns=['Source', 'Target'])
    df_neg_train['Class'] = 0
    df_train = df_train.append(df_neg_train, sort=True)
    df_train= df_train[['Date','Source', 'Target','Class']]
    return df_train

def negative_edge_sampling_test(g, df_train, df_test):
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
        if (s,t) in test_non_edges or (s,t) in train_non_edges:
            continue
        if g.has_edge(s,t):
            continue
        test_non_edges.add((s,t))
    df_neg_test = pd.DataFrame(list(test_non_edges), columns=['Source', 'Target'])
    df_neg_test['Class'] = 0
    df_test = df_test.append(df_neg_test, sort=True)
    df_test= df_test[['Date','Source', 'Target','Class']]
    return df_test

def negative_edge_sampling(g, df_train, df_test):
    df_train = negative_edge_sampling_train(g, df_train)
    df_test = negative_edge_sampling_test(g, df_train, df_test)
    print('size train:',len(df_train))
    print('size test:',len(df_test))
    return df_train, df_test


def test_features(train, test,max_depth=20, gamma=4, scale_pos_weight=1, min_child_weight=1, print_results=True):

    seed = 2
    # Traditional data
    X_train = train.iloc[:, 3:len(train.columns)-1]
    Y_train = train.iloc[:,len(train.columns)-1]
    X_test = test.iloc[:, 3:len(test.columns)-1]
    Y_test = test.iloc[:,len(test.columns)-1]

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

def save_results(filename,model, size, results):
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

def create_file(filename):
    with open(filename, 'w') as file:
        file.write('size,model,precision,recall,roc_auc,accuracy,f1_score\n')



def test_multiple_features(g, df_train, df_test, print_results=False, paths={}, heuristic=True, node2vec=True, deepwalk=True):
    """
    Extract features and run classifiers
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
    Given a train set, extracts test samples usefull for training.
    """   
    g, df_train = create_train_graph(df_train, wcc=wcc)

    # Remove unseen nodes from test
    mask = df_test.apply(lambda x: g.has_node(x['Source']) and g.has_node(x['Target']) and not g.has_edge(x['Source'],x['Target']),axis=1)
    df_test = df_test[mask]
    return g, df_train, df_test


def train_test_sampling(train_path, test_path, method='combined',sampling='node', paths={}, print_results=False, sample_sizes=[], resume=False, wcc=False):
    """
    Train and Test features on different data files.
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
            df_train = df_train_full # sample_graph(df_train_full,sample_size,sampling,g=G)
            df_test = df_test_full #sample_graph(df_test_full,sample_size*test_ratio,sampling,g=G)
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

    with open(results_path,'a') as file:
        for method, data in result.items():
            precision, recall, roc_auc, accuracy, f1 = map(lambda x: round(x,4),data)
            file.write(f'{day},rf,{method},{precision},{recall},{roc_auc},{accuracy},{f1}\n')

        

if __name__ == "__main__":
    # logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
    
    # sample_sizes = [1000000 + 1000000*i for i in range(10)] #1M
    # sample_sizes = [50000 + 50000*i for i in range(30)] #random
    
    # train_test_sampling(
    #     'data/dynamobi/2008-08-01.txt.gz',
    #     'data/dynamobi/2008-08-02.txt.gz',
    #     sample_sizes=sample_sizes,
    #     sampling='random',
    #     wcc=False,
    #     paths={
    #         'heuristic': 'results/dynamobi/15_fullday_heuristic.csv',
    #         'node2vec': 'results/dynamobi/15_fullday_node2vec.csv',
    #         'deepwalk': 'results/dynamobi/15_fullday_deepwalk.csv'
    #     },
    #     print_results=True,
    #     resume=False,
    #     method='separated')
    train_size_full_test(600000,'data/dynamobi/2008-07-28.txt.gz','results/dynamobi/17_train0728_600k.csv')


    print('DONE')