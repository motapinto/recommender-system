import scipy.sparse as sps
from sklearn.utils.extmath import randomized_svd
from Recommenders.Base.BaseItemSimilarityMatrix import BaseItemSimilarityMatrix
from Utils.methods.seconds_to_biggest_unit import seconds_to_biggest_unit
from Utils.methods.compute_w_sparse import compute_W_sparse_from_item_latent_factors

class PureSVDItem(BaseItemSimilarityMatrix):
    RECOMMENDER_NAME = 'PureSVDItem'

    def __init__(self, URM_train, verbose=True):
        super(PureSVDItem, self).__init__(URM_train, verbose=verbose)

    def fit(self, num_factors=24, topK=1462, random_seed=None):
        self._print('Computing SVD decomposition...')
        U, Sigma, QT = randomized_svd(
            self.URM_train,
            n_components=num_factors,
            random_state = random_seed)

        if topK is None:
            topK = self.n_items

        W_sparse = compute_W_sparse_from_item_latent_factors(QT.T, topK=topK)
        self.W_sparse = sps.csr_matrix(W_sparse)
        self._print('Computing SVD decomposition... Done!')
