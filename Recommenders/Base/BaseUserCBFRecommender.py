import numpy as np
from Recommenders.Base.BaseRecommender import BaseRecommender as _BaseRecommender
from Recommenders.recommender_utils import check_matrix

class BaseUserCBFRecommender(_BaseRecommender):
    '''
    This class refers to a BaseRecommender which uses content features, it provides only one function
    to check if users exist that have no features
    '''

    def __init__(self, URM_train, UCM_train, verbose = True):
        super(BaseUserCBFRecommender, self).__init__(URM_train, verbose = verbose)

        assert self.n_users == UCM_train.shape[0], '{}: URM_train has {} users but UCM_train has {}'.format(self.RECOMMENDER_NAME, self.n_items, UCM_train.shape[0])

        self.UCM_train = check_matrix(UCM_train.copy(), 'csr', dtype=np.float32)
        self.UCM_train.eliminate_zeros()

        _, self.n_features = self.UCM_train.shape

        self._cold_user_CBF_mask = np.ediff1d(self.UCM_train.indptr) == 0

        if self._cold_user_CBF_mask.any():
            print('{}: UCM Detected {} ({:4.1f}%) cold users.'.format(
                self.RECOMMENDER_NAME, self._cold_user_CBF_mask.sum(), self._cold_user_CBF_mask.sum()/self.n_users*100))
