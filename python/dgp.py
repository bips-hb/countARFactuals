import networkx as nx
import random
import torch
import pyro.distributions as dist
from functools import partial
import pandas as pd
import numpy as np
import math
import dill
import os
import warnings
import argparse

# functions that aggregate over the event dimension (-1) but leave batch shape etc unaffected
AGG_FNCS = {
    'SUM': lambda pv : pv.sum(-1)
}

def get_random_agg_fnc(d):
    weights = dist.Normal(0.0, 1.0).sample((d,))
    weights = weights / weights.abs().sum()
    bias = dist.Normal(0.0, 0.5).sample((1,))
    fnc = lambda pv: (weights * pv).sum(-1) + bias
    return fnc

# functions that take aggregation function and parent values and return parametrized distribution
DIST_FNCS = {
    'NORMAL': lambda pv, agg: dist.Normal(agg(pv), 2),
    'BERN': lambda pv, agg: dist.Binomial(probs=torch.sigmoid(agg(pv)))
}

def get_random_normal_dist_fnc(agg):
    sd = torch.abs(dist.Normal(0.0, 2.0).sample((1,)))
    mu = dist.Normal(0, 3).sample((1,))
    dist_fnc = lambda pv: dist.Normal(agg(pv)+mu, sd)
    return dist_fnc


class DGP:
    """ A class to represent a broad class of Bayesian Networks.
    The user specifies a directed acyclic graph (DAG) and a set of conditional distributions.
    The class can then be used to sample from the model and to calculate the log likelihood of the data.
    For efficient computation, the computation is parallelized over the batch dimension.
    The batch dimension must be speficied when creating the class but can be changed later.
    """
    def __init__(self, graph, dist_fncs, batch_size):
        """
        Args:
            graph (nx.DiGraph): A directed acyclic graph.
            dist_fncs (dict): A dictionary of conditional distributions, where the keys are the names of the nodes.
            batch_size (int): The batch size.
        """
        self.graph = graph  
        self.nodes = list(nx.topological_sort(self.graph))
        self.fncs = dist_fncs
        self.values = {}
        self.dists = {}
        self.batch_size = batch_size

    @staticmethod
    def generate_graph(d, p=0.5, seed=None):
        if seed is None:
            seed = random.randint(0, 1000)
        graph = nx.gnp_random_graph(d, p, seed, directed=True)
        graph.remove_edges_from([(u, v) for (u, v) in graph.edges() if u > v])
        assert nx.is_directed_acyclic_graph(graph)
        # get max degree node
        degrees = dict(nx.degree(graph))
        y = max(degrees, key=degrees.get)
        # relabel nodes
        nodes = list(nx.topological_sort(graph))
        mapping = {}
        ii = 1
        for node in nodes:
            if node != y:
                mapping[node] = 'x{}'.format(ii)
                ii += 1
            else:
                mapping[node] = 'y'
        graph = nx.relabel_nodes(graph, mapping)
        return graph
    
    @staticmethod
    def generate_model(graph, batch_size=10000):
        dist_fncs = {}
        dgp = DGP(graph, dist_fncs, batch_size)
        cat_nodes = list(np.random.choice(dgp.nodes, size=math.floor(len(dgp.nodes)/2)))
        if 'y' not in cat_nodes:
            cat_nodes.append('y')
        for node in dgp.nodes:
            d = len(dgp.get_parents(node))
            agg_fnc = get_random_agg_fnc(d)
            if node in cat_nodes:
                dist_fncs[node] = partial(DIST_FNCS['BERN'], agg=agg_fnc)
            else:
                dist_fncs[node] = get_random_normal_dist_fnc(agg_fnc)
        dgp = DGP(graph, dist_fncs, batch_size)
        return dgp
        
    def set_batch_size(self, batch_size):
        """Allows changing the batch size.
        
        Args:
            batch_size (int): The new batch size.
        """
        # delete all values if batch size changes
        if self.batch_size != batch_size:
            print('All values reset with batch_size change.')
            for key in self.values.keys():
                self.values[key] = None
        self.batch_size = batch_size

    def set_values(self, values):
        """Sets the values of the nodes. When setting the values the conditional distribution
        objects are updated as well to reflect the changes in parent values.

        Args:
            values: A pandas DataFrame or a torch.Tensor with the values of the nodes.
                (The last dimension must be the same as the number of nodes,
                and the first dimension must be the batch size.)
        """
        # sets values and updates dists
        assert values.shape[-1] == len(self.nodes)
        self.set_batch_size(values.shape[0])
        for node in self.nodes:
            # store values
            if isinstance(values, pd.DataFrame):
                self.values[node] = torch.tensor(values[node])
            elif isinstance(values, torch.Tensor):
                self.values[node] = values[..., self.nodes.index(node)] # set values
            else:
                raise NotImplementedError
            
            # update dists
            self.dists[node] = self.get_dist_node(node) # adapt dists according to parent values

    def get_values(self, as_df=True):
        """Allows retrieving the values as a pd.DataFrame.
        """
        if as_df:
            vals = []
            for node in self.nodes:
                vals.append(self.values[node])
            vals = torch.stack(vals, -1).numpy()
            return pd.DataFrame(vals, columns=self.nodes)
        raise NotImplementedError
        
    def get_parents(self, node):
        """Returns the parents of a node sorted topologically."""
        assert node in self.graph.nodes
        parents = list(self.graph.predecessors(node))
        parents = sorted(parents, key=self.nodes.index)
        return parents
    
    def is_root(self, node):
        """Returns True if the node is a root node."""
        return len(self.get_parents(node)) == 0
    
    def get_parents_values(self, node):
        """Returns the values of the parents of a node."""
        if self.is_root(node):
            return torch.tensor([])
        parents = self.get_parents(node)
        valss = []
        for parent in parents:
            assert not self.values[parent] is None
            valss.append(self.values[parent])
        valss = torch.stack(valss, -1)
        return valss    
    
    def get_dist_node(self, node):
        """Returns the conditional distribution of a node given its parents.
        The parents values are stored in the DGP object and can be updated by 
        calling the set_values method.
        """
        parent_values = self.get_parents_values(node)
        dist = self.fncs[node](parent_values)
        if self.is_root(node):
            dist = dist.expand([self.batch_size])
        assert dist.batch_shape == torch.Size([self.batch_size])
        assert dist.event_shape == torch.Size([])
        self.dists[node] = dist
        return dist

    def sample_node(self, node):
        """Samples a node given the values of its parents.
        The parent values are stored in the DGP object and can be updated using the
        set_values method.
        """
        dist = self.get_dist_node(node)
        sample = dist.sample()
        assert sample.shape == torch.Size([self.batch_size])
        self.values[node] = sample
        return sample

    def sample(self):
        """Samples from the model according to the batch size."""
        samples = []
        for node in nx.topological_sort(self.graph):
            sample = self.sample_node(node)
            samples.append(sample)
        samples = torch.stack(samples, -1)
        assert samples.shape == torch.Size([self.batch_size, len(self.graph.nodes)])
        return samples

    def log_prob_node(self, node):
        """Gets the log prob of a node given the values of its parents.
        The values of node and parents are stored in the DGP 
        and can be updated using the set_values method."""
        log_likelihood = self.dists[node].log_prob(self.values[node])
        return log_likelihood

    def log_prob(self, data, return_df=False):
        """Calculates the log likelihood of the data."""
        self.set_values(data)
        log_probs = []
        for node in self.nodes:
            log_probs.append(self.log_prob_node(node))
        log_prob = torch.stack(log_probs, -1).sum(-1)
        if return_df:
            df = self.get_values(as_df=True)
            df['log_prob'] = log_prob
            return df
        return log_prob
    
    def save(self, path, foldername):
        """Saves the DGP object to a file."""
        if not os.path.exists(path):
            raise ValueError('Path does not exist')
        if not path[-1] == '/':
            raise ValueError('Path must end with /')
        
        savepath = path + foldername + '/'
        if os.path.exists(savepath):
            raise ValueError('Folder already exists. Rename or remove folder. Savepath: ' + savepath)
        
        os.makedirs(path + foldername + '/')
        
        nx.write_gexf(self.graph, savepath + 'graph.gexf')
        with open(savepath + 'dist_fncs.pkl', 'wb') as f:
            dill.dump(self.fncs, f)


    @staticmethod
    def load(path, foldername, batch_size):
        """Loads a DGP object from a file."""
        loadpath = path + foldername + '/'
        if not os.path.exists(loadpath):
            raise ValueError('Folder does not exist')
        
        graph = nx.read_gexf(loadpath + 'graph.gexf')
        with open(loadpath + 'dist_fncs.pkl', 'rb') as f:
            dist_fncs = dill.load(f)
        dgp = DGP(graph, dist_fncs, batch_size)
        return dgp
    

def model_gen(d, p, batch_size, it=0, lim=100):
    graph = DGP.generate_graph(d, p)
    dgp = DGP.generate_model(graph, batch_size)

    dgp.sample()
    values = dgp.get_values(as_df=True)

    # check whether model performance is ok
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = values.sample(5000, replace=True)
    X, y = data.drop('y', axis=1), data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = RandomForestClassifier(n_estimators=100, max_depth=50)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    if 0.7 < accuracy < 0.9:
        print('Model performance is ok.')
        return dgp
    else:
        print('Model performance bad.')
        if it < lim:
            print('Trying again')
            return model_gen(d, p, batch_size, it=it+1)
        else:
            raise RuntimeError('Could not find a good model after {} iterations.'.format(lim))
        
def save_dgp_and_data(dgp, path, dgpname):
    print('trying to save ' + dgpname + ' to ' + path)
    dgp.sample()
    values = dgp.get_values(as_df=True)

    dgp.save(path + 'dgps/', dgpname)
    if not os.path.exists(path + dgpname + '.csv'):
        print('saving values to csv')
        values.to_csv(path + dgpname + '.csv', index=False)
    else:
        warnings.warn('{}.csv already exists.'.format(dgpname))

    # check whether loded DGP gives same likelihoods
    dgp_check = DGP.load(path + 'dgps/', dgpname, dgp.batch_size)
    log_prob_check = dgp_check.log_prob(values, return_df=False)
    log_prob = dgp.log_prob(values, return_df=False)
    assert sum(log_prob != log_prob_check) == 0


## BN_1 graph
def gen_bn_5(batch_size):
    graph = nx.DiGraph()
    graph.add_node('x1')
    graph.add_node('x2')
    graph.add_node('x3')
    graph.add_node('x4')
    graph.add_node('y')

    graph.add_edge('x1', 'x3')
    graph.add_edge('x2', 'x3')
    graph.add_edge('x3', 'x4')
    graph.add_edge('x1', 'x4')
    graph.add_edge('x2', 'x4')
    graph.add_edge('x4', 'y')
    graph.add_edge('x2', 'y')

    # specify the conditional distributions
    dist_fncs = {}
    dist_fncs['x1'] = lambda pv: dist.Binomial(probs=torch.tensor([0.7]))
    dist_fncs['x2'] = lambda pv: dist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    dist_fncs['x3'] = lambda pv: dist.Binomial(probs=torch.sigmoid(pv.sum(-1)))
    dist_fncs['x4'] = lambda pv: dist.Normal(pv.sum(-1), 1.0)
    dist_fncs['y'] = lambda pv: dist.Binomial(probs=torch.sigmoid(pv.sum(-1)))

    # create the DGP object and save the dgp and sampled values
    dgp = DGP(graph, dist_fncs, batch_size)
    return dgp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate and save synthetic data and models.')
    parser.add_argument('--batch_size', type=int, default=100000, help='The batch size.')
    args = parser.parse_args()

    batch_size = args.batch_size

    dgp = gen_bn_5(batch_size)
    try:
        save_dgp_and_data(dgp, 'python/synthetic/', 'bn_5')
    except Exception as e:
        print(e)
        warnings.warn('Could not save bn_5') 


    ## generate random models and save them

    gen_dict = {
        'bn_10' : (10, 0.5),
        'bn_100' : (100, 0.5),
    }

    for name in gen_dict.keys():
        d, p = gen_dict[name]
        dgp = model_gen(d, p, batch_size)
        try:
            save_dgp_and_data(dgp, 'python/synthetic/', name)
        except Exception as e:
            print(e)
            warnings.warn('Could not save {}'.format(name))
