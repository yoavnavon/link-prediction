import random
import networkx as nx

def random_walk_sample(g,size):
    sample = nx.DiGraph()
    nodes = list(g.nodes())
    init_node = random.choice(nodes)
    node = init_node
    steps = 0
    while len(sample.edges()) < size:
        if steps == 100:
            init_node = new_init_node(nodes,g)
            node = init_node
            steps = 0
        neighbors = list(nx.neighbors(g, node))
        #print(neighbors)
        if not neighbors:
            node = init_node
            steps += 1
            continue
        next_node = random.choice(neighbors)
        edge = (node,next_node)
        attr = g.get_edge_data(*edge)
        sample.add_edge(node,next_node,**attr)
        node = next_node
        if random.random() < 0.15:
            init_node = new_init_node(nodes, g)
            node = init_node
    return sample

def new_init_node(nodes, g):
    init_node = random.choice(nodes)
    while not len(list(nx.neighbors(g,init_node))):
        init_node = random.choice(nodes)
    return init_node

def edges_to_df(g, edges):
    subgraph = g.edge_subgraph(edges)
    df = nx.to_pandas_edgelist(subgraph, source='Source', target='Target')
    df = df.drop_duplicates(subset=['Source','Target'])
    df['Class'] = 1
    df['Date']= df['Date'].astype('datetime64[ns]')
    df['Source'] = df['Source'].astype(str)
    df['Target'] = df['Target'].astype(str)
    df = df[['Date', 'Source', 'Target','Class']]
    return df

def node_sampling(g, size):
    print('Node Sampling')
    nodes = list(g.nodes())
    edges = set()
    while len(edges) < size:
        sampled_nodes = random.sample(nodes,1000)
        edges.update(g.out_edges(sampled_nodes))
        edges.update(g.in_edges(sampled_nodes))
    return edges_to_df(g,edges)

def traverse_sampling(g,size,method='bfs'):
    index = 0 if method == 'bfs' else -1
    nodes = list(g.nodes())
    root = random.choice(nodes)
    sampled_edges = set()
    sampled_nodes = set(root)
    node_list = [root]
    while len(sampled_edges) < size:
        if node_list:
            node = node_list.pop(index)
            edges = g.edges(node)
            new_nodes = [t for s,t in edges if s not in sampled_nodes]
            sampled_nodes.update(new_nodes)
            node_list += new_nodes
            sampled_edges.update(edges)
        else:
            node_list.append(random.choice(nodes))
    return edges_to_df(g,sampled_edges)


def bfs_sampling(g, size):
    print('BFS Sampling')
    return traverse_sampling(g,size,method='bfs')
    

def dfs_sampling(g, size):
    print('DFS Sampling')
    return traverse_sampling(g,size,method='dfs')