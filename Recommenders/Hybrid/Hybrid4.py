import os
import numpy as np
from sklearn.preprocessing import normalize
from Recommenders.Base.Base import Base
from Utils.import_recommenders import *

class Hybrid4(Base):
    RECOMMENDER_NAME = 'Hybrid4'
    
    def __init__(self, URM_train, ICM, models_folder='test'):
        super(Hybrid4, self).__init__(URM_train)
    
        output_folder_path = os.path.join('Recommenders', 'saved_models', models_folder+os.sep)
    
        EASE_R_trained = EASE_R(URM_train)
        SLIMElasticNet_trained = MultiThreadSLIM_SLIMElasticNet(URM_train)
        ItemKNNCF_trained = ItemKNNCF(URM_train)

        try:
            EASE_R_trained.load_model(output_folder_path)
            SLIMElasticNet_trained.load_model(output_folder_path)
            ItemKNNCF_trained.load_model(output_folder_path)
        except:
            EASE_R_trained.fit()
            SLIMElasticNet_trained.fit()
            ItemKNNCF_trained.fit()

        self.EASE_R = EASE_R_trained
        self.SLIMElasticNet = SLIMElasticNet_trained
        self.ItemKNNCF = ItemKNNCF_trained

        self.num_train_items = URM_train.shape[1]

    def fit(self, alpha=0.5, beta=0.5, gamma=0.5, delta=0.5, epsilon=0.5, zeta=0.5, eta=0.5, norm='max'):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        self.eta = eta

        self.norm = norm

        # self.theta = theta
        # self.iota = iota

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights = np.zeros([len(user_id_array), self.num_train_items])
        
        ease_r_w = self.EASE_R._compute_item_score(user_id_array, items_to_compute)
        item_knn_cf_w = self.ItemKNNCF._compute_item_score(user_id_array, items_to_compute)
        slim_elastic_net_w = self.SLIMElasticNet._compute_item_score(user_id_array, items_to_compute)
        
        for idx, user in enumerate(user_id_array):
            interactions = len(self.URM_train[user,:].indices)

            if interactions < 128: 
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), norm=self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), norm=self.norm, axis=1)
                w = w1 * self.alpha + w2 * abs(1 - self.alpha)
                
            elif interactions < 200: 
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), norm=self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), norm=self.norm, axis=1)
                w = w1 * self.beta + w2 * abs(1 - self.beta)
            
            elif interactions < 273: 
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), norm=self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), norm=self.norm, axis=1)
                w = w1 * self.gamma + w2 * abs(1 - self.gamma)
            
            elif interactions < 354: 
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), norm=self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), norm=self.norm, axis=1)
                w = w1 * self.delta + w2 * abs(1 - self.delta)
            
            elif interactions < 485: 
                w1 = item_knn_cf_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), norm=self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), norm=self.norm, axis=1)
                w = w1 * self.epsilon + w2 * abs(1 - self.epsilon)
            
            elif interactions < 1733: 
                w1 = ease_r_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), norm=self.norm, axis=1)
                w2 = slim_elastic_net_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), norm=self.norm, axis=1)
                w = w1 * self.zeta + w2 * abs(1 - self.zeta)
            
            else:
                w1 = ease_r_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), norm=self.norm, axis=1)
                w2 = slim_elastic_net_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), norm=self.norm, axis=1)
                w = w1 * self.eta + w2 * abs(1 - self.eta)

            item_weights[idx,:] = w

        return item_weights