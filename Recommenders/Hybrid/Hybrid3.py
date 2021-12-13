import numpy as np
from numpy import linalg as LA
from Recommenders.Base.Base import Base
from Recommenders.Base.TopPop import TopPop
from Recommenders.CF.KNN.ItemKNNCF import ItemKNNCF
from Recommenders.CF.KNN.EASE_R import EASE_R
from Recommenders.CF.MatrixFactorization.PureSVDItem import PureSVDItem

class Hybrid3(Base):
    RECOMMENDER_NAME = 'Hybrid3'
    
    def __init__(self, URM_train, ICM=None):
        super(Hybrid3, self).__init__(URM_train)
        
        self.ICM = ICM

        self.TopPop = TopPop(self.URM_train)
        self.ItemKNNCF = ItemKNNCF(self.URM_train)
        self.EASE_R = EASE_R(self.URM_train)

        self.num_train_items = URM_train.shape[1]

    def fit(self, threshold=10, alpha=0.22053391007450424):
        self.threshold = threshold
        self.alpha = alpha

        self.TopPop.fit()
        self.ItemKNNCF.fit()
        self.EASE_R.fit()

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights = np.zeros([len(user_id_array), self.num_train_items])
        top_pop_w = self.TopPop._compute_item_score(user_id_array, items_to_compute)
        item_knn_cf_w = self.ItemKNNCF._compute_item_score(user_id_array, items_to_compute)
        ease_r_w = self.EASE_R._compute_item_score(user_id_array, items_to_compute)
        
        for idx, user in enumerate(user_id_array):
            interactions = len(self.URM_train[user,:].indices)
            
            # if interactions < self.threshold: 
            #     w = top_pop_w[idx]
            #     w /= LA.norm(w, 2)
            #     item_weights[idx,:] = w

            if interactions > 340 and interactions < 500:
                w = item_knn_cf_w[idx]
                w /= LA.norm(w, 2)
                item_weights[idx,:] = w
                
            else:
                w = ease_r_w[idx]
                w /= LA.norm(w, 2)
                item_weights[idx,:] = w 

        return item_weights



