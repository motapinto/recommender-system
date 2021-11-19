import numpy as np
from Recommenders.recommender_utils import check_matrix
from Recommenders.Base.BaseUserSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender
from Utils.methods.ir_feature_weighting import okapi_BM_25, TF_IDF
from Recommenders.Similarity.Compute_Similarity import Compute_Similarity

class UserKNNCFRecommender(BaseUserSimilarityMatrixRecommender):
    ''' UserKNN recommender'''

    RECOMMENDER_NAME = 'UserKNNCFRecommender'
    FEATURE_WEIGHTING_VALUES = ['BM25', 'TF-IDF', 'none']

    def __init__(self, URM_train, verbose = True):
        super(UserKNNCFRecommender, self).__init__(URM_train, verbose = verbose)

    def fit(self, topK=364, shrink=226, similarity='tanimoto', normalize=True, feature_weighting='none', URM_bias=True, **similarity_args):
        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(f'Value for \'feature_weighting\' not recognized. Acceptable values are {self.FEATURE_WEIGHTING_VALUES}, provided was {feature_weighting}')

        if URM_bias is not None:
            self.URM_train.data += URM_bias

        if feature_weighting == 'BM25':
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == 'TF-IDF':
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        similarity = Compute_Similarity(self.URM_train.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
