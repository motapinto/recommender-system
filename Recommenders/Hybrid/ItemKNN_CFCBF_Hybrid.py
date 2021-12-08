import numpy as np
import scipy.sparse as sps
from Recommenders.CB.KNN.ItemKNNCBF import ItemKNNCBF

class ItemKNN_CFCBF_Hybrid(ItemKNNCBF):
    RECOMMENDER_NAME = 'ItemKNN_CFCBF_Hybrid'

    def fit(self, ICM_weight = 1.0, **fit_args):

        if(len(fit_args) == 0):
            fit_args = {
                'topK': 71,
                'shrink': 881,
                'similarity': 'asymmetric',
                'normalize': True,
                'asymmetric_alpha': 0.07813729267797666,
                'feature_weighting': 'TF-IDF',
            }

            ICM_weight = 0.07306776133242769

        self.ICM_train = self.ICM_train*ICM_weight
        self.ICM_train = sps.hstack([self.ICM_train, self.URM_train.T], format='csr')

        super(ItemKNN_CFCBF_Hybrid, self).fit(**fit_args)

    def _get_cold_item_mask(self):
        return np.logical_and(self._cold_item_CBF_mask, self._cold_item_mask)
