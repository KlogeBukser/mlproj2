class Hyperparams:
    def __init__(self):
        self.learning_rates = [0.02,0.3]
        self.batch_vals = [1,20]
        self.lmbdas = [-8,-1]
        self.eta_algos = ['basic','ada','rms','adam']
        self.params = {'learning_rates' : self.learning_rates, 'batch_vals' : self.batch_vals, 'lmbdas' : self.lmbdas}

    def __call__(self,rng):
        return rng.uniform(*self.params['learning_rates']), rng.integers(*self.params['batch_vals']), 10**rng.uniform(*self.params['lmbdas']), rng.choice(self.eta_algos)
        
        
    def rm_algo(self,algo):
        self.eta_algos.remove(algo)

    def change_limits(self,param,new_limit,upper = True):
        if upper:
            self.params[param][1] = new_limit
        else:
            self.params[param][0] = new_limit