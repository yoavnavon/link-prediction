import pandas as pd
import numpy as np
import networkx as nx
import random
import math
from datetime import datetime, timedelta
from utils import move_class_col

from node2vec import Node2Vec
from gensim.models import Word2Vec
from node2vec.edges import HadamardEmbedder
import sys
sys.path.append('./GraphEmbedding')
from ge import Node2Vec, DeepWalk

def get_heuristics(g, df_train, df_test):
    # Compute heuristic features
    df_trad_train = df_train.copy()
    df_trad_test = df_test.copy()
    df_trad = df_trad_train.append(df_trad_test)

    # print('propflow')
    # df_trad['propflow']= df_trad.Source.apply(lambda x: propflow(g, x,5)) 
    # print('shortest path ')
    # df_trad['shortest_path_all_nodes']= df_trad.Source.apply(lambda x: len(list(nx.single_source_shortest_path(g, x, cutoff=None)))-1) 
    
    print('katz')
    d = nx.katz_centrality(g)
    temp = pd.DataFrame.from_dict(d, orient='index', columns = ['katz']) #move dict of degrees to dataframe
    temp['Source'] = temp.index #add index as source
    df_trad= pd.merge(df_trad,temp, how='left', on= 'Source')  #merge frames
    print('degree')
    df_trad['in_degree_i_to_j'] = df_trad.Source.apply(g.in_degree) 
    df_trad['out_degree_i_to_j']= df_trad.Source.apply(g.out_degree) 
    df_trad['in_degree_j_to_i']= df_trad.Target.apply(g.in_degree) 
    df_trad['out_degree_j_to_i'] = df_trad.Target.apply(g.out_degree) 
    print('common nbrs')
    df_trad['common_nbrs']= df_trad.apply(lambda x: len(set(g.neighbors(x['Source'])).intersection(set(g.neighbors(x['Target'])))), axis=1) 
    df_trad['volume_i']= df_trad.apply(lambda x: nx.volume(g, (x['Source'], x['Target'])), axis=1) 
    df_trad['volume_j']= df_trad.apply(lambda x: nx.volume(g, (x['Target'], x['Source'])), axis=1) 
    print('similarities')
    df_trad['adamic_adar']= df_trad.apply(lambda x: similarity(g, x['Source'], x['Target'], method = "adamic_adar"), axis=1) 
    df_trad['pref_attach']= df_trad.apply(lambda x: similarity(g, x['Source'], x['Target'], method = "preferential_attachment"), axis=1) 
    df_trad['jaccards_coef']= df_trad.apply(lambda x: similarity(g, x['Source'], x['Target'], method = "jaccard"), axis=1)

    df_trad = move_class_col(df_trad,'Class')
    df_trad = df_trad.fillna(0)
    df_trad_train = df_trad.iloc[:len(df_trad_train)]
    df_trad_test = df_trad.iloc[len(df_trad_train):]
    return df_trad_train, df_trad_test

def get_node2vec(g, df_train, df_test, p=1, q=1):
    node2vec = Node2Vec(g, walk_length=10, num_walks=80, p=p, q=q, workers=5)
    node2vec.train(window=4, iter=4, workers=5)
    emb = node2vec.get_embeddings()

    node2vec_hadd_train = df_train.apply(lambda x: manual_haddamard(x,emb), axis= 1)
    node2vec_hadd_test = df_test.apply(lambda x: manual_haddamard(x,emb), axis= 1)

    df_node2vec_train = pd.concat([df_train, node2vec_hadd_train],axis=1)
    df_node2vec_train = move_class_col(df_node2vec_train, 'Class')
    df_node2vec_test = pd.concat([df_test, node2vec_hadd_test],axis=1)
    df_node2vec_test = move_class_col(df_node2vec_test, 'Class')

    return df_node2vec_train, df_node2vec_test

def get_deepwalk(g, df_train, df_test):
    model = DeepWalk(g, walk_length=10, num_walks=80, workers=5)
    model.train(window_size=4, iter=4, workers=5)
    emb = model.get_embeddings()

    deepwalk_hadd_train = df_train.apply(lambda x: manual_haddamard(x,emb), axis= 1)
    deepwalk_hadd_test = df_test.apply(lambda x: manual_haddamard(x,emb), axis= 1)

    df_deepwalk_train = pd.concat([df_train, deepwalk_hadd_train],axis=1)
    df_deepwalk_train = move_class_col(df_deepwalk_train, 'Class')
    df_deepwalk_test = pd.concat([df_test, deepwalk_hadd_test],axis=1)
    df_deepwalk_test = move_class_col(df_deepwalk_test, 'Class')
    return df_deepwalk_train, df_deepwalk_test


def shortest_path_count(x):
    return len(list(nx.all_shortest_paths(g, x['Source'], x['Target'], weight=1))) 

def shortest_path_all_nodes(x):
    return len(list(nx.single_source_shortest_path(g, x['Source'], cutoff=None)))-1
     
def katz( x):
    return katz_similarity(g, x['Source'], x['Target'])
            
def max_flow( x):
    return (nx.maximum_flow(g, x['Source'], x['Target'])[0])

def prop(x):
    l = 5
    return propflow(g, x['Source'],l)

def propflow(Graph, root, l):
    scores = {}
    
    n1 = root
    found = [n1]
    newSearch = [n1]
    scores[n1]=1.0
    
    for currentDegree in range(0,l+1):
        oldSearch = list(newSearch)
        newSearch = []
        while len(oldSearch) != 0:
            n2 = oldSearch.pop()
            nodeInput = scores[n2]
            sumOutput = 0.0
            #Node2 = Graph.GetNI(n2)
            for n3 in Graph.edges(n2):
                if Graph.get_edge_data(n2,n3) is None:
                    continue
                else:
                    sumOutput += Graph.get_edge_data(n2,n3)["weight"]
            flow = 0.0
            for n3 in Graph.edges(n2):
                wij = Graph.get_edge_data(n2,n3)
                if Graph.get_edge_data(n2,n3) is None:
                    flow = 0
                else:
                    flow = nodeInput * (wij*1.0/sumOutput)
                if n3 not in scores:
                    scores[n3]=0.0
                scores[n3] += flow
                if n3 not in found:
                    found.append(n3)
                    newSearch.append(n3)
    return np.mean(list(scores.values()))

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

def manual_haddamard(x,emb):
    """
    Computed haddamard between two embeddings.
    """
    return pd.Series(emb[x['Source']]*emb[x['Target']])
