import numpy as np
from sklearn.preprocessing import normalize
from Recommenders.Hybrid.ItemKNNSimilarityHybrid import ItemKNNSimilarityHybrid
from Recommenders.recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.Base.BaseItemSimilarityMatrix import BaseItemSimilarityMatrix
from Utils.methods.ir_feature_weighting import okapi_BM_25, TF_IDF
from Recommenders.Similarity.Compute_Similarity import Compute_Similarity

class ItemKNNCF(BaseItemSimilarityMatrix):
    RECOMMENDER_NAME = 'ItemKNNCF'
    FEATURE_WEIGHTING_VALUES = ['BM25', 'TF-IDF', 'none']

    def __init__(self, URM_train, verbose=False):
        super(ItemKNNCF, self).__init__(URM_train, verbose=verbose)

    def fit(self, topK=42, shrink=27, alpha=0.85, feature_weighting='BM25', URM_bias=1000, tversky_alpha=0.0246, tversky_beta=1.1590, asymmetric_alpha=0.5):
        self.topK = topK
        self.shrink = shrink

        self.URM_train = normalize(self.URM_train, norm='l2', axis=1)

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

        sim1 = Compute_Similarity(self.URM_train, shrink=shrink, topK=topK, normalize=True, similarity='tversky', tversky_alpha=tversky_alpha, tversky_beta=tversky_beta)
        sim2 = Compute_Similarity(self.URM_train, shrink=shrink, topK=topK, normalize=True, similarity='asymmetric', asymmetric_alpha=asymmetric_alpha)

        self.W_sparse = sim1.compute_similarity()*alpha + sim2.compute_similarity()*(1-alpha)
        self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
