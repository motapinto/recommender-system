
import numpy as np
from tqdm import tqdm
from numpy import linalg as LA
from Recommenders.Base.Base import Base
from Recommenders.CF.KNN.ItemKNNCF import ItemKNNCF
from Recommenders.CF.KNN.UserKNNCF import UserKNNCF
from Recommenders.CB.KNN.ItemKNNCBF import ItemKNNCBF
from Recommenders.CB.KNN.UserKNNCBF import UserKNNCBF

class Hybrid(Base):
  RECOMMENDER_NAME = 'Hybrid'
  def __init__(self, URM_train, ICM):
    super(Hybrid, self).__init__(URM_train)
    
    self.ICM = ICM
    self.ItemCF = ItemKNNCF(self.URM_train)
    self.UserCF = UserKNNCF(self.URM_train)
    # self.ItemCBF = ItemKNNCBF(self.URM_train, self.ICM)
    # self.UserCBF = UserKNNCBF(self.URM_train, self.ICM)

    self.num_train_items = URM_train.shape[1]

  def fit(self):
    # ICM = similaripy.normalization.bm25plus(self.ICM.copy())
    # URM_aug = sps.vstack([self.URM_train, ICM.T])
    # URM_aug2 = sps.vstack([self.URM_train, self.ICM.T])
    # URM_aug2 = similaripy.normalization.bm25plus(URM_aug2)

    # Fit the recommenders
    self.ItemCF.fit()
    self.UserCF.fit()
    # self.ItemCBF.fit()
    # self.UserCBF.fit()

  def _compute_item_score(self, user_id_array, items_to_compute=None):
    item_weights = np.empty([len(user_id_array), self.num_train_items])
    
    for i in tqdm(range(len(user_id_array))):
      interactions = len(self.URM_train[user_id_array[i],:].indices)
      if interactions > 0:
        w1 = self.ItemCF._compute_item_score(user_id_array[i], items_to_compute) 
        w1 /= LA.norm(w1, 2)

        w2 = self.UserCF._compute_item_score(user_id_array[i], items_to_compute)
        w2 /= LA.norm(w2, 2) 
        
        # w3 = self.ItemCBF._compute_item_score(user_id_array[i], items_to_compute)
        # w3 /= LA.norm(w3, 2)
        
        # w4 = self.UserCBF._compute_item_score(user_id_array[i], items_to_compute)
        # w4 /= LA.norm(w4, 2)
        
        w = w1 + w2 #+ w3 + w4
        item_weights[i,:] = w

    return item_weights

