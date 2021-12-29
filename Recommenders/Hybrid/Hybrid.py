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

    def fit(self, alpha=0.9, beta=0.95, gamma=0.65, delta=0.5, epsilon=0.55, zeta=0.3, norm='max',
        imp1=0.48, imp2=0.7, imp4=0.8, imp5=0.8, imp6=0.8, imp7=0.8
    ):   
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        
        self.norm = norm

        self.imp1 = imp1
        self.imp2 = imp2
        self.imp4 = imp4
        self.imp5 = imp5
        self.imp6 = imp6
        self.imp7 = imp7

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights = np.zeros([len(user_id_array), self.num_train_items])
        
        ease_r_w = self.EASE_R._compute_item_score(user_id_array, items_to_compute)
        slim_elastic_net_w = self.SLIMElasticNet._compute_item_score(user_id_array, items_to_compute)
        item_knn_cf_w = self.ItemKNNCF._compute_item_score(user_id_array, items_to_compute)
        user_knn_cf_w = self.UserKNNCF._compute_item_score(user_id_array, items_to_compute)

        for idx, user in enumerate(user_id_array):
            interactions = len(self.URM_train[user,:].indices)

            if interactions < 128: 
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = item_knn_cf_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)
                w4 = user_knn_cf_w[idx]
                if np.any(w4): w4 = normalize(w4.reshape(1, -1), self.norm, axis=1)
                
                w = self.imp1*(w1*(self.alpha) + w2*(1-self.alpha)) + (1-self.imp1)*(w3*0.5 + w4*0.5)
                
            elif interactions < 200: 
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = item_knn_cf_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)

                w = self.imp2*(w1*self.beta + w2*(1-self.beta))+ w3*(1-self.imp2)
            
            elif interactions < 273:
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                
                w = w1*self.gamma + w2*(1-self.gamma)  
                         
            elif interactions < 354: 
                w1 = slim_elastic_net_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = item_knn_cf_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)

                w = self.imp4*(w1*self.delta + w2*(1-self.delta)) + (1-self.imp4)*w3
            
            elif interactions < 485: 
                w1 = item_knn_cf_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = ease_r_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = slim_elastic_net_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)

                w = w1*self.epsilon + (1-self.epsilon)*(0.5*w2+0.5*w3)
                w *= 0.8
            
            else: 
                w1 = ease_r_w[idx]
                if np.any(w1): w1 = normalize(w1.reshape(1, -1), self.norm, axis=1)
                w2 = slim_elastic_net_w[idx]
                if np.any(w2): w2 = normalize(w2.reshape(1, -1), self.norm, axis=1)
                w3 = item_knn_cf_w[idx]
                if np.any(w3): w3 = normalize(w3.reshape(1, -1), self.norm, axis=1)
                
                w = w1*self.zeta + (1-self.zeta)*(0.5*w2+0.5*w3)
                w *= 0.8

            item_weights[idx,:] = w

        return item_weights