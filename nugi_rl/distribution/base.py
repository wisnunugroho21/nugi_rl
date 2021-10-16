class Distribution():
    def sample(self, datas):
        raise NotImplementedError
        
    def entropy(self, datas):
        raise NotImplementedError
        
    def logprob(self, datas, value_data):
        raise NotImplementedError

    def kldivergence(self, datas1, datas2):
        raise NotImplementedError

    def deterministic(self, datas):
        raise NotImplementedError