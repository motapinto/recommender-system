import numpy as np
from numpy import linalg as LA
from Recommenders.Base.Base import Base
from Recommenders.CF.KNN.P3alpha import P3alpha
from Recommenders.CF.KNN.ItemKNNCF import ItemKNNCF
from Recommenders.CF.KNN.UserKNNCF import UserKNNCF

class Hybrid1(Base):
    RECOMMENDER_NAME = 'Hybrid1'
    
    def __init__(self, URM_train, ICM):
        super(Hybrid1, self).__init__(URM_train)
        
        self.ICM = ICM

        self.P3alpha = P3alpha(self.URM_train)
        self.ItemKNNCF = ItemKNNCF(self.URM_train)
        self.UserKNNCF = UserKNNCF(self.URM_train)

        self.num_train_items = URM_train.shape[1]

    def fit(self, threshold=11, alpha=0.5):
        self.threshold = threshold
        self.alpha = alpha

        self.P3alpha.fit()
        self.ItemKNNCF.fit()
        self.UserKNNCF.fit()

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights = np.empty([len(user_id_array), self.num_train_items])
        p3_alpha_w = self.P3alpha._compute_item_score(user_id_array, items_to_compute)
        item_knn_cf_w = self.ItemKNNCF._compute_item_score(user_id_array, items_to_compute)
        user_knn_cf_w = self.UserKNNCF._compute_item_score(user_id_array, items_to_compute)
        
        for idx, user in enumerate(user_id_array):
            interactions = len(self.URM_train[user,:].indices)
            
            #  P3-alpha has a high popularity bias
            if interactions < self.threshold: 
                w = p3_alpha_w[idx]
                w /= LA.norm(w, 2)
                item_weights[idx,:] = w
                
            else:                    
                w1 = item_knn_cf_w[idx]
                w1 /= LA.norm(w1, 2)

                w2 = user_knn_cf_w[idx]
                w2 /= LA.norm(w2, 2)

                item_weights[idx,:] = w1 * (self.alpha) + w2 * (1-self.alpha)

        return item_weights



