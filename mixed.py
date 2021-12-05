from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

import numpy as np

class Mixed(BaseRecommender):
    
    RECOMMENDER_NAME = "Mixed"
    
    def __init__(self, URM_train, ICMge, ICMsg, ICMch):
        super(Mixed, self).__init__(URM_train)
        self.ICMge = ICMge
        self.ICMsg = ICMsg
        self.ICMch = ICMch
    
    def fit(self, shrink=0, topK=100, w1=1/3,w2=1/3,w3=1/3):
        
        self.w1=w1
        self.w2=w2
        self.w3=w3
        
        self.CBFge = ItemKNNCBFRecommender(URM_train, ICMge)
        self.CBFsg = ItemKNNCBFRecommender(URM_train, ICMsg)
        self.CBFch = ItemKNNCBFRecommender(URM_train, ICMch)
        
        self.CBFge.fit(shrink=shrink, topK=topK)
        self.CBFsg.fit(shrink=shrink, topK=topK)
        self.CBFch.fit(shrink=shrink, topK=topK)
        
        
    def _compute_item_score(self, user_id_array, items_to_compute = None, ):
        item_weights = np.empty([len(user_id_array), 18059])
        for i in range(len(user_id_array)):
            s1 = self.CBFge._compute_item_score(user_id_array[i], items_to_compute)
            s2 = self.CBFsg._compute_item_score(user_id_array[i], items_to_compute)
            s3 = self.CBFch._compute_item_score(user_id_array[i], items_to_compute)
            score = s1*self.w1+s2*self.w2+s3*self.w3
            item_weights[i,:] = score 

        return item_weights