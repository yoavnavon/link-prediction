import random
import networkx as nx
import math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix

def timestamp():
    """
    Create timestamp from current datetime in seconds
    """
    now = datetime.now()
    return str(datetime.timestamp(now)).split('.')[0]

def random_date(start, end):
    """
    This function will return a random datetime between two datetime 
    objects.
    """
    
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

def run_model(model, x, y):
    modell= type(model).__name__
    print(modell)
    num_folds = 2
    seed = 2
    kfold = StratifiedKFold(n_splits=num_folds, random_state=seed)    
    results = cross_val_score(model, x, y, cv=kfold)
    auc= cross_val_score(model, x, y, cv=kfold, scoring='roc_auc')
    print(('accuracy = {} acc_std = {}'.format(results.mean()*100.0, results.std()*100.0)))
    print('auc = {} auc_std = {}'.format(auc.mean()*100.0, auc.std()*100.0))

    
def run_model_test(model,train_x, train_y, test_x, test_y, class_weights=None):
    
    modell= type(model).__name__
    print(modell)
    model.fit(train_x, train_y)
    # scores = model.get_booster().get_score(importance_type="gain")
    predicted = model.predict(test_x)
    y_pred_prob = model.predict_proba(test_x)[:,1]
    matrix = confusion_matrix(test_y, predicted)
    print(matrix)
    precision = precision_score(test_y, predicted)
    recall = recall_score(test_y, predicted)
    roc_auc = roc_auc_score(test_y, predicted)
    accuracy = accuracy_score(test_y, predicted)
    f1 = f1_score(test_y, predicted)
    # if modell == 'RandomForestClassifier':
    #     print(train_x.columns)
    #     print(model.feature_importances_)
    return precision, recall, roc_auc, accuracy, f1


def get_randomwalk(g, node, path_length):
    random_walk = [str(node)]
    for i in range(path_length-1):
        temp = list(g.neighbors(node))
        temp = list(set(temp) - set(random_walk))    
        if len(temp) == 0:
            break
        random_node = random.choice(temp)
        random_walk.append(str(random_node))
        node = random_node
    return random_walk

def get_randomwalks(g, n_walks=5, path_length=10):
    all_nodes = list(g.nodes())
    random_walks = []
    for n in tqdm(all_nodes):
        for i in range(n_walks):
            random_walks.append(get_randomwalk(g, n, path_length))
    return random_walks

def random_walk_stats(random_walks):
    """
    Returns the number of sentences (walks) for each length
    """
    stats = defaultdict(lambda: 0)
    for i in random_walks:
        stats[len(i)] += 1
    return stats

def manual_haddamard(x,emb):
    """
    Computed haddamard between two embeddings.
    """
    return pd.Series(emb[x['Source']]*emb[x['Target']])
    # return pd.Series(emb.get_vector(x['Source'])*emb.get_vector(x['Target']))

   
def katz_similarity(katzDict,i,j):
    l = 1
    neighbors = katzDict[i]
    score = 0
    maxl= 2
    while l <= maxl:
        numberOfPaths = nx.degree(j)
        if numberOfPaths > 0:
            score += (beta**l)*numberOfPaths

        neighborsForNextLoop = []
        for k in neighbors:
            neighborsForNextLoop += katzDict[k]
        neighbors = neighborsForNextLoop
        l += 1
    return score


def move_class_col(dfr, column_to_move):
    """moves class column to end.

    Args:
    dfr: A dataframe
    column_to_move: column you want to move to the end

    Returns:
    rearranged dataframe with column at end."""
    cols = list(dfr.columns.values) 
    cols.pop(cols.index(column_to_move)) 
    dfr = dfr[cols+[column_to_move]] 
    return dfr

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



