import os
import numpy as np
from sklearn.preprocessing import normalize
from Recommenders.Base.Base import Base
from Utils.import_recommenders import *

class Hybrid(Base):
    RECOMMENDER_NAME = 'Hybrid'
    
    def __init__(self, URM_train, ICM, models_folder='test'):
        super(Hybrid, self).__init__(URM_train)
        
        output_folder_path = os.path.join('Recommenders', 'saved_models', models_folder+os.sep)
    
        EASE_R_trained = EASE_R(URM_train)
        SLIMElasticNet_trained = MultiThreadSLIM_SLIMElasticNet(URM_train)
        ItemKNNCF_trained = ItemKNNCF(URM_train)
        UserKNNCF_trained = UserKNNCF(URM_train)

        try:
            EASE_R_trained.load_model(output_folder_path)
            SLIMElasticNet_trained.load_model(output_folder_path)
            ItemKNNCF_trained.load_model(output_folder_path)
            UserKNNCF_trained.load_model(output_folder_path)
        except:
            EASE_R_trained.fit()
            SLIMElasticNet_trained.fit()
            ItemKNNCF_trained.fit()
            UserKNNCF_trained.fit()

        self.EASE_R = EASE_R_trained
        self.SLIMElasticNet = SLIMElasticNet_trained
        self.ItemKNNCF = ItemKNNCF_trained
        self.UserKNNCF = UserKNNCF_trained

        self.num_train_items = URM_train.shape[1]

    def fit(self, norm='max'):   
        self.norm = norm

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights = np.zeros([len(user_id_array), self.num_train_items])
        
        ease_r_w = self.EASE_R._compute_item_score(user_id_array, items_to_compute)
        slim_elastic_net_w = self.SLIMElasticNet._compute_item_score(user_id_array, items_to_compute)
        item_knn_cf_w = self.ItemKNNCF._compute_item_score(user_id_array, items_to_compute)
        user_knn_cf_w = self.UserKNNCF._compute_item_score(user_id_array, items_to_compute)

        for idx, user in enumerate(user_id_array):
            interactions = len(self.URM_train[user,:].indices)

            if interactions < 90: # group 0
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = item_knn_cf_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)
                w4 = user_knn_cf_w[idx]
                if np.any(w4): w4 = normalize(w4.reshape(1, -1), self.norm, axis=1)
                
                alpha = 0.47
                beta = 0.92
                gamma = 0.50
                w = alpha*(beta*w1 + (1-beta)*w2) + (1-alpha)*(gamma*w3 + (1-gamma)*w4)
                
                # Test other combination of weights
                #w = alpha*w1 + (1-alpha)*(beta*w2 + (1-beta)*(gamma*w3 + (1-gamma)*w4)
                # w = ...
                # w = ...
                # w = ...

            elif interactions < 144: # group 1
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = item_knn_cf_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)
                w4 = user_knn_cf_w[idx]
                if np.any(w4): w4 = normalize(w4.reshape(1, -1), self.norm, axis=1)
                
                alpha = 0.48
                beta = 0.94
                gamma = 0.46
                w = alpha*(beta*w1 + (1-beta)*w2) + (1-alpha)*(gamma*w3 + (1-gamma)*w4)
                
            elif interactions < 186: # group 2
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = item_knn_cf_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)
                w4 = user_knn_cf_w[idx]
                if np.any(w4): w4 = normalize(w4.reshape(1, -1), self.norm, axis=1)
                
                alpha = 0.47
                beta = 0.98
                gamma = 0.45
                w = alpha*(beta*w1 + (1-beta)*w2) + (1-alpha)*(gamma*w3 + (1-gamma)*w4)
            
            elif interactions < 228: # group 3
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = item_knn_cf_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)
                w4 = user_knn_cf_w[idx]
                if np.any(w4): w4 = normalize(w4.reshape(1, -1), self.norm, axis=1)
                
                alpha = 0.47
                beta = 0.95
                gamma = 0.45
                w = alpha*(beta*w1 + (1-beta)*w2) + (1-alpha)*(gamma*w3 + (1-gamma)*w4)
            
            elif interactions < 272: # group 4
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = item_knn_cf_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)
                w4 = user_knn_cf_w[idx]
                if np.any(w4): w4 = normalize(w4.reshape(1, -1), self.norm, axis=1)
                
                alpha = 0.65
                beta = 0.94
                gamma = 0.45  
                w = alpha*(beta*w1 + (1-beta)*w2) + (1-alpha)*(gamma*w3 + (1-gamma)*w4)

            elif interactions < 319: # group 5
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = item_knn_cf_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)
                w4 = user_knn_cf_w[idx]
                if np.any(w4): w4 = normalize(w4.reshape(1, -1), self.norm, axis=1)
                
                alpha = 0.69
                beta = 0.98
                gamma = 0.5  
                w = alpha*(beta*w1 + (1-beta)*w2) + (1-alpha)*(gamma*w3 + (1-gamma)*w4)
            
            elif interactions < 373: # group 6 
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = item_knn_cf_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)
                w4 = user_knn_cf_w[idx]
                if np.any(w4): w4 = normalize(w4.reshape(1, -1), self.norm, axis=1)
                
                alpha = 0.65
                beta = 0.91
                gamma = 0.44
                w = alpha*(beta*w1 + (1-beta)*w2) + (1-alpha)*(gamma*w3 + (1-gamma)*w4)
            
            elif interactions < 452: # group 7
                w1 = ease_r_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = item_knn_cf_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = slim_elastic_net_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)
                
                alpha = 0.3
                beta = 0.6
                w = alpha*w1 + (1-alpha)*(beta*w2 + (1-beta)*w3)

            elif interactions < 577: # group 8
                w1 = ease_r_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = item_knn_cf_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = slim_elastic_net_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)
                
                alpha = 0.31
                beta = 0.6
                w = alpha*w1 + (1-alpha)*(beta*w2 + (1-beta)*w3)
            
            else: # group 9
                w1 = ease_r_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = slim_elastic_net_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = item_knn_cf_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)
                
                alpha = 0.3
                beta = 0.55
                w = alpha*w1 + (1-alpha)*(beta*w2 + (1-beta)*w3)

            item_weights[idx,:] = w

        return item_weights

# MAP = 0.25404