import numpy as np
from Recommenders.recommender_utils import check_matrix
from Recommenders.Base.BaseItemSimilarityMatrix import BaseItemSimilarityMatrix
from Utils.methods.ir_feature_weighting import okapi_BM_25, TF_IDF
from Recommenders.Similarity.Compute_Similarity import Compute_Similarity

class ItemKNNCF(BaseItemSimilarityMatrix):
    RECOMMENDER_NAME = 'ItemKNNCF'
    FEATURE_WEIGHTING_VALUES = ['BM25', 'TF-IDF', 'none']

    def __init__(self, URM_train, verbose=True):
        super(ItemKNNCF, self).__init__(URM_train, verbose=verbose)

    def fit(self, topK=46, shrink=31, similarity='tversky', normalize=True, feature_weighting='BM25', URM_bias=1000.0, **similarity_args):
        self.topK = topK
        self.shrink = shrink

        if(len(similarity_args) == 0):
            similarity_args = {
                'tversky_alpha': 0.022632810966513674, 
                'tversky_beta': 1.0589896897155855,
            } 

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

        similarity = Compute_Similarity(self.URM_train, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
