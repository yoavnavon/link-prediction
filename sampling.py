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


def node_sampling(g, size):
    print('Node Sampling')
    nodes = list(g.nodes())
    edges = set()
    while len(edges) < size:
        sampled_nodes = random.sample(nodes,100)
        for node in sampled_nodes:
            edges.update(g.edges(node))

    subgraph = g.edge_subgraph(edges)
    df = nx.to_pandas_edgelist(subgraph, source='Source', target='Target')
    df = df.drop_duplicates(subset=['Source','Target'])
    df['Class'] = 1
    df['Date']= df['Date'].astype('datetime64[ns]')
    df['Source'] = df['Source'].astype(str)
    df['Target'] = df['Target'].astype(str)
    df = df[['Date', 'Source', 'Target','Class']]
    return df