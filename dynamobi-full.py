import pandas as pd
import numpy as np
import networkx as nx
import random
import math
from multiprocessing import  Pool
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

from node2vec import Node2Vec
from gensim.models import Word2Vec
from node2vec.edges import HadamardEmbedder
import sys
sys.path.append('./GraphEmbedding')
from ge import Node2Vec, DeepWalk
from utils import *
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


def create_train_test_split(df_clean):
    split = 0.5
    # g = nx.from_pandas_edgelist(df_clean, source='Source', target='Target', create_using=nx.DiGraph())   #put dataframe to an edgelist
    # wcc = max(nx.weakly_connected_components(g), key=len)
    # g = g.subgraph(wcc)
    # df_clean = df_clean[df_clean.apply(lambda x: g.has_edge(x['Source'],x['Target']),axis=1)] # reduce df with subgraph

    print('Spliting')
    df_train = df_clean.iloc[:int(len(df_clean)*split)]
    df_test = df_clean.iloc[int(len(df_clean)*split):]
    print('Creating Graph')
    g = nx.from_pandas_edgelist(df_train, source='Source', target='Target', create_using=nx.DiGraph())
    print(nx.info(g))
    
    print('Removing Unseen Nodes from Test')
    mask = df_test['Source'].apply(g.has_node) & df_test['Target'].apply(g.has_node)
    df_test = df_test[mask]
    return g, df_train, df_test

def negative_edge_sampling(g, df_train, df_test):
    # Sample negative edges
    nodes = list(g.nodes())
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




def common_nbrs(x):
    return len(set(g.neighbors(x['Source'])).intersection(set(g.neighbors(x['Target']))))

def shortest_path_count(x):
    return len(list(nx.all_shortest_paths(g, x['Source'], x['Target'], weight=1))) 

def shortest_path_all_nodes(x):
    return len(list(nx.single_source_shortest_path(g, x['Source'], cutoff=None)))-1

def degree_i_to_j(x):
    return (g.degree(x['Source']))

def degree_j_to_i(x):
    return (g.degree(x['Target']))

def in_degree_i_to_j(x):
    return (g.in_degree(x['Source']))
    
def out_degree_i_to_j(x):
    return (g.out_degree(x['Source']))

def in_degree_j_to_i(x):
    return (g.in_degree(x['Target']))
    
def out_degree_j_to_i(x):
    return (g.out_degree(x['Target']))

def volume_j(x):
    return (nx.volume(g, (x['Target'], x['Source'])))

def volume_i(x):
    return (nx.volume(g, (x['Source'], x['Target'])))

def max_flow(x):
    pass
    
def katz( x):
    return katz_similarity(g, x['Source'], x['Target'])

def jaccard( x):
    return similarity(g, x['Source'], x['Target'], method = "jaccard")

def pref_attach(x):
    return similarity(g, x['Source'], x['Target'], method = "preferential_attachment")

def adamic( x):
    return similarity(g, x['Source'], x['Target'], method = "adamic_adar")
            
def max_flow( x):
    return (nx.maximum_flow(g, x['Source'], x['Target'])[0])

def prop(x):
    l = 5
    return propflow(g, x['Source'],l)

def similarity(graph, i, j, method):
    if method == "common_neighbors":
        return len(set(graph.neighbors(i)).intersection(set(graph.neighbors(j))))
    elif method == "jaccard":
        if float(len(set(graph.neighbors(i)).union(set(graph.neighbors(j))))) == 0:
            return 0
        else:
            return len(set(graph.neighbors(i)).intersection(set(graph.neighbors(j))))/float(len(set(graph.neighbors(i)).union(set(graph.neighbors(j)))))
    elif method == "adamic_adar":
        sum_score = 0
        for v in set(graph.neighbors(i)).intersection(set(graph.neighbors(j))):
            degree = graph.degree(v)
            score = 0 if degree == 1 else 1.0/math.log(graph.degree(v))
            sum_score += score
        return sum_score
    elif method == "preferential_attachment":
        return graph.degree(i) * graph.degree(j)
    elif method == "friendtns":
        return round((1.0/(graph.degree(i) + graph.degree(j) - 1.0)),3)
 
def get_heuristics(g, df_train, df_test):
    # Compute heuristic features
    df_trad_train = df_train.copy()
    df_trad_test = df_test.copy()
    df_trad = df_trad_train.append(df_trad_test)

    # print('propflow')
    # df_trad['propflow']= df_trad.apply(prop, axis=1) 
    # df_trad['shortest_path_all_nodes']= df_trad.apply(shortest_path_all_nodes, axis=1) 
    
    d = nx.katz_centrality(g)
    temp = pd.DataFrame.from_dict(d, orient='index', columns = ['katz']) #move dict of degrees to dataframe
    temp['Source'] = temp.index #add index as source
    df_trad= pd.merge(df_trad,temp, how='left', on= 'Source')  #merge frames
    df_trad['in_degree_i_to_j'] = df_trad.apply(in_degree_i_to_j, axis=1) 
    df_trad['out_degree_i_to_j']= df_trad.apply(out_degree_i_to_j, axis=1) 
    df_trad['in_degree_j_to_i']= df_trad.apply(in_degree_j_to_i, axis=1) 
    df_trad['out_degree_j_to_i'] = df_trad.apply(out_degree_j_to_i, axis=1) 
    df_trad['common_nbrs']= df_trad.apply(common_nbrs, axis=1) 
    df_trad['volume_i']= df_trad.apply(volume_i, axis=1) 
    df_trad['volume_j']= df_trad.apply(volume_j, axis=1) 
    df_trad['adamic_adar']= df_trad.apply(adamic, axis=1) 
    df_trad['pref_attach']= df_trad.apply(pref_attach, axis=1) 
    df_trad['jaccards_coef']= df_trad.apply(jaccard, axis=1)

    df_trad = move_class_col(df_trad,'Class')
    df_trad = df_trad.fillna(0)
    df_trad_train = df_trad.iloc[:len(df_trad_train)]
    df_trad_test = df_trad.iloc[len(df_trad_train):]
    return df_trad_train, df_trad_test

def get_node2vec(g, df_train, df_test, p=1, q=1):
    node2vec = Node2Vec(g, walk_length=10, num_walks =100, p=p, q=q, workers=4)
    node2vec.train(window = 4, iter = 4, workers = 4)
    emb = node2vec.get_embeddings()

    node2vec_hadd_train = df_train.apply(lambda x: manual_haddamard(x,emb), axis= 1)
    node2vec_hadd_test = df_test.apply(lambda x: manual_haddamard(x,emb), axis= 1)

    df_node2vec_train = pd.concat([df_train, node2vec_hadd_train],axis=1)
    df_node2vec_train = move_class_col(df_node2vec_train, 'Class')
    df_node2vec_test = pd.concat([df_test, node2vec_hadd_test],axis=1)
    df_node2vec_test = move_class_col(df_node2vec_test, 'Class')

    return df_node2vec_train, df_node2vec_test

def get_deepwalk(g, df_train, df_test):
    model = DeepWalk(g, walk_length=10, num_walks=100, workers=4)
    model.train(window_size=4, iter=4, workers=4)
    emb = model.get_embeddings()

    deepwalk_hadd_train = df_train.apply(lambda x: manual_haddamard(x,emb), axis= 1)
    deepwalk_hadd_test = df_test.apply(lambda x: manual_haddamard(x,emb), axis= 1)

    df_deepwalk_train = pd.concat([df_train, deepwalk_hadd_train],axis=1)
    df_deepwalk_train = move_class_col(df_deepwalk_train, 'Class')
    df_deepwalk_test = pd.concat([df_test, deepwalk_hadd_test],axis=1)
    df_deepwalk_test = move_class_col(df_deepwalk_test, 'Class')
    return df_deepwalk_train, df_deepwalk_test

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

def test_features_on_size(file_paths, sample_sizes, sampling='time', save=False, p=1, q=1):
    """
    Reads data file, splits it in train/test, then test features
    """
    
    global g
    heuristic_results_path = file_paths[0]
    node2vec_results_path = file_paths[1]
    deepwalk_results_path = file_paths[2]
    create_file(heuristic_results_path)
    create_file(node2vec_results_path)
    create_file(deepwalk_results_path)

    df = read_file('clean/2008-07-30.txt.gz')
    G = None
    if sampling == 'node':
        G = nx.from_pandas_edgelist(df, source='Source', target='Target',edge_attr='Date', create_using=nx.DiGraph())
    
    for sample_size in sample_sizes:
        df_clean = sample_graph(df, sample_size, sampling, save, g=G)
        g, df_train, df_test = create_train_test_split(df_clean)
        df_train, df_test = negative_edge_sampling(g, df_train, df_test)
        test_multiple_features(g, df_train, df_test, print_result=True) 


def run_single_test(size=20000,sampling='time',save=False,file='', p=1, q=1):
    global g
    G = None
    df = read_file(file)
    if sampling == 'node':
        G = nx.from_pandas_edgelist(df, source='Source', target='Target',edge_attr='Date', create_using=nx.DiGraph())
    df_clean = sample_graph(df, size, sampling, save, g=G)
    g, df_train, df_test = create_train_test_split(df_clean)
    df_train, df_test = negative_edge_sampling(g, df_train, df_test)
    test_multiple_features(g, df_train, df_test, print_result=True) 

def run_node2vec_gridsearch(sample_size=20000,save=False, sampling='node'):
    global g

    df = read_file('clean/2008-07-29.txt.gz')
    G = None
    if sampling == 'node':
        G = nx.from_pandas_edgelist(df, source='Source', target='Target',edge_attr='Date', create_using=nx.DiGraph())
    
    for p in [0.25,1,2]:
        for q in [1]:
            df_clean = sample_graph(df, sample_size, sampling, save, g=G)
            g, df_train, df_test = create_train_test_split(df_clean)
            df_train, df_test = negative_edge_sampling(g, df_train, df_test)
            
            df_node2vec_train, df_node2vec_test = get_node2vec(g, df_train, df_test,p=p,q=q)
            print('Node2Vec p:{}, q:{}'.format(p,q))
            size, xgb_results_n2v, rf_results_n2v = test_features(df_node2vec_train, df_node2vec_test, max_depth=20, gamma=4, scale_pos_weight=1, min_child_weight=1)
            print_results('xgb', size, xgb_results_n2v)
            print_results('rf', size, rf_results_n2v)

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
            save_results(paths[method],'xgb', size, xgb)
            save_results(paths[method],'rf', size, rf)
        if print_result:
            print(method)
            print_results('xgb', size, xgb)
            print_results('rf', size, rf)



def train_test_node_sampling(train_path, test_path, method='combined',sampling='node', paths={}, print_result=False, sample_sizes=[], resume=False):
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
            df_sample = sample_graph(df_data,sample_size,sampling,False,g=G)
            df_train = df_sample[df_sample.Date.dt.day == train_day]
            df_test = df_sample[df_sample.Date.dt.day == test_day]
        
        print('Creating Graph')
        g = nx.from_pandas_edgelist(df_train, source='Source', target='Target', create_using=nx.DiGraph())
        print(nx.info(g))
        
        print('Removing Unseen Nodes from Test')
        mask = df_test['Source'].apply(g.has_node) & df_test['Target'].apply(g.has_node)
        df_test = df_test[mask]

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
    # pandarallel.initialize(nb_workers=12)
    g = None
    sample_sizes = [25000 + 25000*i for i in range(30)]
    
    train_test_node_sampling(
        'clean/2008-07-28.txt.gz',
        'clean/2008-07-29.txt.gz',
        sample_sizes=sample_sizes,
        sampling='random',
        paths={
            'heuristic': 'results/random_wcc_sampling_heuristic.csv',
            'node2vec': 'results/random_wcc_sampling_node2vec.csv',
            'deepwalk': 'results/random_wcc_sampling_deepwalk.csv'
        },
        print_result=False,
        resume=False)


    print('DONE')