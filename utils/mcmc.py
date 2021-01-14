import numpy as np
import pandas as pd
from scipy.optimize import minimize
np.random.seed(223)

class Cfg_Opts(object):
    def __init__(self,):
        self.csv_file     = './work_sapce/csv/val_prd.csv'
        self.num_class    = 32094
        self.max_iters    = 100#000
        self.max_epochs   = 50#00

class Optimizer(object):
    def __init__(self, cfg):
        self.num_class   = cfg.num_class
        self.max_iters   = cfg.max_iters
        self.max_epochs  = cfg.max_epochs

        self.csv_data    = pd.read_csv(cfg.csv_file)
        self.gt_labels   = self.csv_data['gt'].values
        
        self.class_names = []
        for key in self.csv_data.keys():
            if(('gt' not in key) and (('pd') not in key)):
                self.class_names.append(key)

        self.prob_datas = [] ### [num_class, m_samples]
        for i in range(len(self.class_names)):
            self.prob_datas.append(np.array(self.csv_data[self.class_names[i]]))
        self.prob_datas = np.array(self.prob_datas)
        self.prob_datas = self.prob_datas.T  ### [m_samples, num_class]
        
        self.all_loss    = []
        self.all_weights = []
        
    def optimize(self):
        for i in range(self.max_epochs):
            starting_values = np.random.uniform(size=self.num_class)
            #cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
            bounds = [(0,1)]*self.num_class
            res = minimize(self.loss_fun(), starting_values, method='L-BFGS-B', bounds=bounds, options={'disp': False, 'maxiter': self.max_iters})

            self.all_loss.append(res['fun'])
            self.all_weights.append(res['x'])
            
            print('Score: {score}'.format(score=res['fun']))

    # def loss_fun(self):
        # def optimize_func(weights):
            # ''' scipy minimize will pass the weights as a numpy array '''
            
            # weights = np.tile(weights, (len(self.prob_datas),1))
            # tmp_prob_datas = weights*self.prob_datas
            # all_predicts = np.argmax(tmp_prob_datas, axis=1)
            
            # comatch = all_predicts==self.gt_labels
            # correct = np.where(comatch)
            # corrnum = len(correct[0])
            # score = 1- corrnum/len(all_predicts)
            # return  score 
            
        # return optimize_func
        
    def loss_fun(self):
        def optimize_func(weights):
            ''' scipy minimize will pass the weights as a numpy array '''
            all_predicts = []
            for predicts in self.prob_datas:
               # print(prediction.shape, weights.shape)
                weight_predict = weights*predicts
                pd_l = np.argmax(weight_predict)
                all_predicts.append(pd_l)

           # print(final_prediction.shape)
            all_predicts = np.array(all_predicts)
            comatch = all_predicts==self.gt_labels
            correct = np.where(comatch)
            corrnum = len(correct[0])
            score   = 1- corrnum/len(all_predicts)
            return  score 
            
        return optimize_func

    def get_optimized(self):
        bestSC   = np.min(self.all_loss)
        bestWght = self.all_weights[np.argmin(self.all_loss)]

        print('\n Ensemble Score: {best_score}'.format(best_score=bestSC))
        print('\n Best Weights: {weights}'.format(weights=bestWght))

if __name__ == '__main__':
    cfg = Cfg_Opts()  
    trainer = Optimizer(cfg)
    trainer.optimize()
    trainer.get_optimized()
    