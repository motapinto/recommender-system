
import numpy as np
from tqdm import tqdm
from numpy import linalg as LA
from Recommenders.Base.Base import Base
from Recommenders.Base.TopPop import TopPop
from Recommenders.CF.KNN.ItemKNNCF import ItemKNNCF
from Recommenders.CF.KNN.UserKNNCF import UserKNNCF
from Recommenders.CF.KNN.EASE_R import EASE_R

class Hybrid2(Base):
    RECOMMENDER_NAME = 'Hybrid2'
    
    def __init__(self, URM_train, ICM):
        super(Hybrid2, self).__init__(URM_train)
        
        self.ICM = ICM

        self.TopPop = TopPop(self.URM_train)
        self.ItemKNNCF = ItemKNNCF(self.URM_train)
        self.EASE_R = EASE_R(self.URM_train)

        self.num_train_items = URM_train.shape[1]

    def fit(self, threshold=5):
        self.threshold = threshold

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
            
            if interactions < self.threshold: 
                w = top_pop_w[idx]
                w /= LA.norm(w, 2)
                item_weights[idx,:] = w
                
            else:
                w1 = item_knn_cf_w[idx]
                w1 /= LA.norm(w1, 2)

                w2 = ease_r_w[idx]
                w2 /= LA.norm(w2, 2)

                item_weights[idx,:] = w1 + w2

        return item_weights



