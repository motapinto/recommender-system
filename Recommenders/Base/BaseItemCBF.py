import numpy as np
from Recommenders.Base.Base import Base as _Base
from Recommenders.recommender_utils import check_matrix

class BaseItemCBF(_Base):
    '''
    This class refers to a Base which uses content features, it provides only one function
    to check if items exist that have no features
    '''

    def __init__(self, URM_train, ICM_train, verbose=True):
        super(BaseItemCBF, self).__init__(URM_train, verbose=verbose)

        assert self.n_items == ICM_train.shape[0], '{}: URM_train has {} items but ICM_train has {}'.format(self.RECOMMENDER_NAME, self.n_items, ICM_train.shape[0])
        
        self.ICM_train = check_matrix(ICM_train.copy(), 'csr', dtype=np.float32)
        self.ICM_train.eliminate_zeros()
        
        _, self.n_features = self.ICM_train.shape

        self._cold_item_CBF_mask = np.ediff1d(self.ICM_train.indptr) == 0

        if self._cold_item_CBF_mask.any():
            print('{}: ICM Detected {} ({:4.1f}%) items with no features.'.format(
                self.RECOMMENDER_NAME, self._cold_item_CBF_mask.sum(), self._cold_item_CBF_mask.sum()/self.n_items*100))
