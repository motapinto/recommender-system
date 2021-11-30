import scipy.sparse as sps
from Recommenders.Base.BaseLightFM import BaseLightFM
from Recommenders.Base.BaseItemCBF import BaseItemCBF

class LightFMItemHybrid(BaseItemCBF, BaseLightFM):
    RECOMMENDER_NAME = 'LightFMItemHybrid'

    def __init__(self, URM_train, ICM_train, verbose=True):
        super(LightFMItemHybrid, self).__init__(URM_train, ICM_train, verbose=verbose)
        self.UCM_train = None

        # Need to hstack item_features to ensure each ItemIDs are present in the model
        eye = sps.eye(self.n_items, self.n_items).tocsr()
        self.ICM_train = sps.hstack((eye, self.ICM_train)).tocsr()
