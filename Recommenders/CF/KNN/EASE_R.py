import time
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
from Utils.methods.seconds_to_biggest_unit import seconds_to_biggest_unit
from Recommenders.Similarity.Compute_Similarity import Compute_Similarity
from Recommenders.recommender_utils import similarityMatrixTopK, check_matrix
from Recommenders.Base.BaseItemSimilarityMatrix import BaseItemSimilarityMatrix

class EASE_R(BaseItemSimilarityMatrix):
    RECOMMENDER_NAME = 'EASE_R'

    def __init__(self, URM_train, sparse_threshold_quota=None):
        super(EASE_R, self).__init__(URM_train)
        self.sparse_threshold_quota = sparse_threshold_quota

    def fit(self, topK=None, l2_norm=3907, verbose=False):
        self.verbose = verbose

        # Grahm matrix is X^t X, compute dot product
        similarity = Compute_Similarity(self.URM_train, shrink=0, topK=self.URM_train.shape[1], normalize=False, similarity='cosine')
        grahm_matrix = similarity.compute_similarity().toarray()
        
        #sim1 = Compute_Similarity(self.URM_train, shrink=0, topK=self.URM_train.shape[1], normalize=True, similarity='tversky', tversky_alpha=1.85, tversky_beta=1.350)
        #sim2 = Compute_Similarity(self.URM_train, shrink=0, topK=self.URM_train.shape[1], normalize=True, similarity='asymmetric', asymmetric_alpha=0.75)
        #grahm_matrix = (sim1.compute_similarity()).toarray()

        #print(similarity.compute_similarity().toarray().shape)
        # print(sim1.compute_similarity().toarray().shape)
        # print(sim2.compute_similarity().toarray().shape)
        
        diag_indices = np.diag_indices(grahm_matrix.shape[0])

        # The Compute_Similarity object ensures the diagonal of the similarity matrix is zero
        # in this case we need the diagonal as well, which is just the item popularity
        item_popularity = np.ediff1d(self.URM_train.tocsc().indptr)
        grahm_matrix[diag_indices] = item_popularity + l2_norm

        P = np.linalg.inv(grahm_matrix)
        B = P / (-np.diag(P))
        B[diag_indices] = 0.0

        # Check if the matrix should be saved in a sparse or dense format
        # The matrix is sparse, regardless of the presence of the topK, if nonzero cells are less than sparse_threshold_quota %
        if topK is not None:
            B = similarityMatrixTopK(B, k = topK, verbose=False)

        if self._is_content_sparse_check(B):
            self._print('Detected model matrix to be sparse, changing format.')
            self.W_sparse = check_matrix(B, format='csr', dtype=np.float32)

        else:
            self.W_sparse = check_matrix(B, format='npy', dtype=np.float32)
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense

    def _is_content_sparse_check(self, matrix):
        if self.sparse_threshold_quota is None:
            return False

        if sps.issparse(matrix):
            nonzero = matrix.nnz
        else:
            nonzero = np.count_nonzero(matrix)

        return nonzero / (matrix.shape[0]**2) <= self.sparse_threshold_quota

    def _compute_score_W_dense(self, user_id_array, items_to_compute = None):
        self._check_format()
        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse)#.toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse)#.toarray()

        return item_scores

    def load_model(self, folder_path, file_name = None):
        super(EASE_R, self).load_model(folder_path, file_name = file_name)

        if not sps.issparse(self.W_sparse):
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense