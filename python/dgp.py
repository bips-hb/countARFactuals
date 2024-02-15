import networkx as nx
import random
import torch
import pyro.distributions as dist
import pandas as pd


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
    def generate_model(graph=None):
        raise NotImplementedError
        
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



# generate graph 
    
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

dgp = DGP(graph, dist_fncs, 10000)
values = dgp.sample() # returns torch tensor
values = dgp.get_values() # returns data frame
values.to_csv('python/synthetic/bn_1.csv',)

# dgp.log_prob_node('x1')
# dgp.log_prob_node('x3')
# df_likelihood = dgp.log_prob(values, return_df=True)
# df_likelihood.to_csv('python/synthetic/bn_1.csv', index=False)

# print(df_likelihood.head())

# import seaborn as sns
import matplotlib.pyplot as plt

# sns.pairplot(df_likelihood, vars=['x1', 'x2', 'x3', 'x4'], hue='y')
# plt.show()

graph = DGP.generate_graph(20, 0.5)
nx.draw(graph, with_labels=True)
plt.show()