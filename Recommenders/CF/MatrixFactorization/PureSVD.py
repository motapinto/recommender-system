import time
import numpy as np
import scipy.sparse as sps
from sklearn.utils.extmath import randomized_svd
from Recommenders.Base.BaseMatrixFactorization import BaseMatrixFactorization
from Utils.methods.seconds_to_biggest_unit import seconds_to_biggest_unit

class PureSVD(BaseMatrixFactorization):
    RECOMMENDER_NAME = 'PureSVD'

    def __init__(self, URM_train, verbose=True):
        super(PureSVD, self).__init__(URM_train, verbose=verbose)

    def fit(self, num_factors=24, random_seed=None):
        start_time = time.time()
        self._print('Computing SVD decomposition...')

        U, Sigma, QT = randomized_svd(
            self.URM_train,
            n_components=num_factors,
            #n_iter=5,
            random_state = random_seed)

        U_s = U * sps.diags(Sigma)

        self.USER_factors = U_s
        self.ITEM_factors = QT.T

        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
        self._print('Computing SVD decomposition... done in {:.2f} {}'.format(new_time_value, new_time_unit))

class ScaledPureSVD(PureSVD):
    RECOMMENDER_NAME = 'ScaledPureSVD'

    def __init__(self, URM_train, verbose=True):
        super(ScaledPureSVD, self).__init__(URM_train, verbose=verbose)


    def fit(self, num_factors=24, random_seed=None, scaling_items=1.0, scaling_users=1.0):
        item_pop = np.ediff1d(sps.csc_matrix(self.URM_train).indptr)
        scaling_matrix = sps.diags(np.power(item_pop, scaling_items - 1))

        self.URM_train = self.URM_train * scaling_matrix

        user_pop = np.ediff1d(sps.csr_matrix(self.URM_train).indptr)
        scaling_matrix = sps.diags(np.power(user_pop, scaling_users - 1))

        self.URM_train = scaling_matrix * self.URM_train

        super(ScaledPureSVD, self).fit(num_factors = num_factors, random_seed = random_seed)
