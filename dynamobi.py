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


def sample_graph(df, size, sampling, save, g=None):
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

    if save:
        df.to_csv(save, header=False)

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


def negative_edge_sampling(g, df_train, df_test):
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

    # source = list(df_test.Source.values[:(len(df_test)//2)]) + random.choices(nodes, k=len(df_test)//2)
    # target = random.choices(nodes, k=len(df_test)//2) + list(df_test.Target.values[:(len(df_test)//2)])
    source = random.choices(nodes, k=len(df_test))
    target = random.choices(nodes, k=len(df_test))

    test_non_edges = set()
    for s,t in zip(source,target):
        if s == t:
            continue
        if (s,t) in test_non_edges or (s,t) in train_non_edges:
            continue
        if g.has_edge(s,t):
            continue
        test_non_edges.add((s,t))

    df_neg_train = pd.DataFrame(list(train_non_edges), columns=['Source', 'Target'])
    df_neg_test = pd.DataFrame(list(test_non_edges), columns=['Source', 'Target'])
    df_neg_train['Class'] = 0
    df_neg_test['Class'] = 0
    df_train = df_train.append(df_neg_train, sort=True)
    df_test = df_test.append(df_neg_test, sort=True)
    df_train= df_train[['Date','Source', 'Target','Class']]
    df_test= df_test[['Date','Source', 'Target','Class']]
    print('size train:', len(df_train))
    print('size test:', len(df_test))
    return df_train, df_test


def test_features(train, test,max_depth=20, gamma=3, scale_pos_weight=2, min_child_weight=5):

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
    xgb_model=XGBClassifier(
        n_estimators=13, 
        max_depth=max_depth,
        gamma=gamma,
        scale_pos_weight=scale_pos_weight,
        min_child_weight= min_child_weight,
        seed=seed,
        predictor='gpu_predictor',nthread=12)
    rf = RandomForestClassifier(n_estimators= 13, max_depth=max_depth,n_jobs=-1)
    xgb_results = run_model_test(xgb_model,X_train, Y_train, X_test, Y_test) 
    rf_results = run_model_test(rf,X_train, Y_train, X_test, Y_test) 
    return len(X_train), xgb_results, rf_results

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



def test_multiple_features(g, df_train, df_test, print_result=False, paths={}, heuristic=True, node2vec=True, deepwalk=True):
    """
    Extract features and run classifiers
    """
    
    results = {}
    
    if heuristic:
        print('Heuristic Features')
        df_trad_train, df_trad_test = get_heuristics(g, df_train, df_test)
        size, xgb_results_trad, rf_results_trad = test_features(df_trad_train, df_trad_test, max_depth=20, gamma=4, scale_pos_weight=1, min_child_weight=1)
        results['heuristic'] = (size, xgb_results_trad, rf_results_trad)
    
    if node2vec:
        print('Node2Vec Embeddings')
        df_node2vec_train, df_node2vec_test = get_node2vec(g, df_train, df_test, p=1, q=1)
        size, xgb_results_n2v, rf_results_n2v = test_features(df_node2vec_train, df_node2vec_test, max_depth=20, gamma=4, scale_pos_weight=1, min_child_weight=1)
        results['node2vec'] = (size, xgb_results_n2v, rf_results_n2v)

    if deepwalk:
        print('Deepwalk Embeddings')
        df_deepwalk_train, df_deepwalk_test = get_deepwalk(g, df_train, df_test)
        size, xgb_results_dw, rf_results_dw = test_features(df_deepwalk_train, df_deepwalk_test, max_depth=20, gamma=4, scale_pos_weight=1, min_child_weight=1)
        results['deepwalk'] = (size, xgb_results_dw, rf_results_dw)

    for method,data in results.items():
        size, xgb, rf = data
        if paths:
            save_results(paths[method],'xgb', size//2, xgb)
            save_results(paths[method],'rf', size//2, rf)
        if print_result:
            print(method)
            print_results('xgb', size//2, xgb)
            print_results('rf', size//2, rf)

def create_train_graph(df_train, wcc=False):
    g = nx.from_pandas_edgelist(df_train, source='Source', target='Target', create_using=nx.DiGraph()) 
    if wcc:
        wcc = max(nx.weakly_connected_components(g), key=len)
        g = g.subgraph(wcc)
        df_train = df_train[df_train.apply(lambda x: g.has_edge(x['Source'],x['Target']),axis=1)] # reduce df with subgraph

    print(nx.info(g))
    return g, df_train


def filter_test(df_train, df_test, wcc=False):    
    g, df_train = create_train_graph(df_train, wcc=wcc)

    # Remove unseen nodes from test
    mask = df_test['Source'].apply(g.has_node) & df_test['Target'].apply(g.has_node)
    df_test = df_test[mask]
    return g, df_train, df_test

def train_test_sampling(train_path, test_path, method='combined',sampling='node', paths={}, print_result=False, sample_sizes=[], resume=False, wcc=False,test_ratio=1):
    """
    Train and Test features on different data files.
    """
    
    global g

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
        if sampling == 'time':
            df_train = sample_graph(df_train_full,sample_size,sampling,False)
            df_test = sample_graph(df_test_full,sample_size,sampling,False)
        else:
            if method == 'combined':
                df_sample = sample_graph(df_data,sample_size,sampling,False,g=G)
                df_train = df_sample[df_sample.Date.dt.day == train_day]
                df_test = df_sample[df_sample.Date.dt.day == test_day]
                g, df_train, df_test = filter_test(df_train, df_test, wcc=wcc)
            elif method == 'separate':
                df_train = sample_graph(df_train_full,sample_size,sampling,False,g=G)
                df_test = sample_graph(df_test_full,sample_size*test_ratio,sampling,False,g=G)
                print(len(df_train),len(df_test))
                g, df_train, df_test = filter_test(df_train, df_test, wcc=wcc)
        

        print('Negative Sampling')
        df_train, df_test = negative_edge_sampling(g, df_train, df_test)
        test_multiple_features(
            g,
            df_train,
            df_test,
            paths=paths,
            print_result=print_result,
            heuristic=True,
            node2vec=True,
            deepwalk=True)
            

        

if __name__ == "__main__":
    # logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
    g = None
    sample_sizes = [25000 + 50000*i for i in range(30)] # random wcc
    
    train_test_sampling(
        'clean/2008-07-28.txt.gz',
        'clean/2008-07-29.txt.gz',
        sample_sizes=sample_sizes,
        sampling='node',
        wcc=False,
        paths={
            # 'heuristic': 'results/node_negsamp2_sampling_heuristic.csv',
            # 'node2vec': 'results/node_negsamp2_sampling_node2vec.csv',
            # 'deepwalk': 'results/node_negsamp2_sampling_deepwalk.csv'
        },
        print_result=False,
        resume=False,
        test_ratio=2,
        method='combined')


    print('DONE')