


class DGP:
    def __init__(self, d, graph=None, model=None):
        self.d = d
        # set graph, generate one if not available
        if graph is None:
            self.graph = self._generate_graph()
        else:
            self.graph = graph
        
        # set model, generate one if not available
        if model is None:
            self.model = self._generate_model()
        else:
            self.model = model

    def _generate_graph(self):
        raise NotImplementedError
    
    def _generate_model(self):
        raise NotImplementedError
    
    def generate_data(self, n):
        raise NotImplementedError
    
    def get_likelyhood(self, n):
        raise NotImplementedError
    


    
