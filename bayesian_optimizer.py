from functools import partial
import numpy as np
import pickle
import GPyOpt


class BayesianOptimizer(object):

    def __init__(self):
        self.params = []
        self.bo = None

    def add_param(self, name, _type, domain, cast=None):
        self.params += [{
            'name': name,
            'type': _type,
            'domain': domain if _type in ['discrete', 'continuous'] else tuple(range(len(domain))),
            'domain_v': domain if _type is 'categorical' else None,
            'dimensionality': 1,
            'cast': cast
        }]

    def get_parameters_values(self, X):
        p_values = dict()
        for i, x in enumerate(X[-1]):
            p = self.params[i]
            cast = p['cast'] if p['cast'] is not None else lambda v: v
            p_values[p['name']] = cast(x if p['type'] != 'categorical' else p['domain_v'][int(x)])
        return p_values

    def create_bo(self, f, maximize=True, initial_design_numdata=5, acquisition_type='EI', acquisition_weight = 0.1, exact_feval=True):
        def func(_X, _f):
            result = _f(**(self.get_parameters_values(_X)))
            print('value={} for params: {}'.format(str(result), self.get_parameters_values(_X)))
            print("*"*30)
            return result

        self.bo = GPyOpt.methods.BayesianOptimization(f=partial(func, _f=f),
                                                      domain=self.params,
                                                      initial_design_numdata=initial_design_numdata,
                                                      acquisition_type=acquisition_type,
                                                      acquisition_weight = acquisition_weight,
                                                      exact_feval=exact_feval,
                                                      maximize=maximize)
        #self.run_opt()

    def run_opt(self, max_iter = 20, max_time = np.inf,  eps = 1e-6):
        return self.bo.run_optimization(max_iter, eps=eps, max_time=max_time)

    def save_report(self, filename):
        bo = self.bo
        assert(bo is not None)
        with open(filename, 'w', encoding='utf8') as f:
            f.write('***** Bayesian optimization *****\n')
            f.write('Parameters settings:\n')
            f.write(str(self.params))
            params, score = bo.get_evaluations()
            best_i = np.argmin(score)
            best_score = score[best_i][0] * (-1 if bo.maximize else 1)
            f.write("\n*********************************\n")
            f.write('Best score: {}\n'.format(str(best_score)))
            f.write('Best params: {}\n'.format(str(self.get_parameters_values([params[best_i]]))))
            f.write("*********************************\n")
            f.write("\n")
            f.write("Runs:\n")
            for i, s in enumerate(score):
                best_score = s[0] * (-1 if bo.maximize else 1)
                f.write('{}: {}\n'.format(str(best_score), str(self.get_parameters_values([params[i]]))))

    def plot_convergence(self):
        self.bo.plot_convergence()
        
    def save_state(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)
            
    def load_state(self, filename):
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict) 
