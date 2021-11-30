import numpy as np
from Recommenders.recommender_utils import check_matrix
from Recommenders.Base.BaseItemCBF import BaseItemCBF
from Recommenders.Base.BaseItemSimilarityMatrix import BaseItemSimilarityMatrix
from Utils.methods.ir_feature_weighting import okapi_BM_25, TF_IDF
from Recommenders.Similarity.Compute_Similarity import Compute_Similarity

class ItemKNNCBF(BaseItemCBF, BaseItemSimilarityMatrix):
    RECOMMENDER_NAME = 'ItemKNNCBF'
    FEATURE_WEIGHTING_VALUES = ['BM25', 'TF-IDF', 'none']

    def __init__(self, URM_train, ICM_train, verbose=True):
        super(ItemKNNCBF, self).__init__(URM_train, ICM_train, verbose=verbose)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting='none', ICM_bias=None, **similarity_args):
        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(f'Value for \'feature_weighting\' not recognized. Acceptable values are {self.FEATURE_WEIGHTING_VALUES}, provided was {feature_weighting}')

        if ICM_bias is not None:
            self.ICM_train.data += ICM_bias

        if feature_weighting == 'BM25':
            self.ICM_train = self.ICM_train.astype(np.float32)
            self.ICM_train = okapi_BM_25(self.ICM_train)

        elif feature_weighting == 'TF-IDF':
            self.ICM_train = self.ICM_train.astype(np.float32)
            self.ICM_train = TF_IDF(self.ICM_train)

        similarity = Compute_Similarity(self.ICM_train.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
