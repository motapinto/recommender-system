from Recommenders.recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.Base.BaseItemSimilarityMatrix import BaseItemSimilarityMatrix

class ItemKNNSimilarityHybrid(BaseItemSimilarityMatrix):
    RECOMMENDER_NAME = 'ItemKNNSimilarityHybrid'

    def __init__(self, URM_train, Similarity_1, Similarity_2, verbose=True):
        super(ItemKNNSimilarityHybrid, self).__init__(URM_train, verbose=verbose)

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError('ItemKNNSimilarityHybrid: similarities have different size, S1 is {}, S2 is {}'.format(
                Similarity_1.shape, Similarity_2.shape
            ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')

    def fit(self, topK=100, alpha = 0.5):
        self.topK = topK
        self.alpha = alpha

        # Hybrid of two similarities S = S1*alpha + S2*(1-alpha)
        W_sparse = self.Similarity_1*self.alpha + self.Similarity_2*(1-self.alpha)

        self.W_sparse = similarityMatrixTopK(W_sparse, k=self.topK)
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
