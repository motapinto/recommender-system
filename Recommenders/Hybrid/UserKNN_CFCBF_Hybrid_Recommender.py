import numpy as np
import scipy.sparse as sps
from Recommenders.Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
from Recommenders.Base.UserKNNCBFRecommender import UserKNNCBFRecommender

class UserKNN_CFCBF_Hybrid_Recommender(UserKNNCBFRecommender, BaseSimilarityMatrixRecommender):
    ''' UserKNN_CFCBF_Hybrid_Recommender'''

    RECOMMENDER_NAME = 'UserKNN_CFCBF_Hybrid_Recommender'

    def fit(self, UCM_weight = 1.0, **fit_args):
        self.UCM_train = self.UCM_train*UCM_weight
        self.UCM_train = sps.hstack([self.UCM_train, self.URM_train], format='csr')

        super(UserKNN_CFCBF_Hybrid_Recommender, self).fit(**fit_args)


    def _get_cold_user_mask(self):
        return np.logical_and(self._cold_user_CBF_mask, self._cold_user_mask)
