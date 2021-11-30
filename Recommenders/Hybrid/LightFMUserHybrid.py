import scipy.sparse as sps
from Recommenders.Base.BaseLightFM import BaseLightFM
from Recommenders.Base.BaseUserCBF import BaseUserCBF

class LightFMUserHybrid(BaseUserCBF, BaseLightFM):
    RECOMMENDER_NAME = 'LightFMUserHybrid'

    def __init__(self, URM_train, UCM_train, verbose=True):
        super(LightFMUserHybrid, self).__init__(URM_train, UCM_train, verbose=verbose)
        self.ICM_train = None

        # Need to hstack user_features to ensure each UserIDs are present in the model
        eye = sps.eye(self.n_users, self.n_users).tocsr()
        self.UCM_train = sps.hstack((eye, self.UCM_train)).tocsr()
