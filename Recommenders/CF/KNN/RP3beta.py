import time, sys
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
from Recommenders.recommender_utils import check_matrix, similarityMatrixTopK
from Utils.methods.seconds_to_biggest_unit import seconds_to_biggest_unit
from Recommenders.Base.BaseItemSimilarityMatrix import BaseItemSimilarityMatrix

class RP3beta(BaseItemSimilarityMatrix):
    RECOMMENDER_NAME = 'RP3beta'

    def __init__(self, URM_train, verbose=True):
        super(RP3beta, self).__init__(URM_train, verbose=verbose)

    def __str__(self):
        return 'RP3beta(alpha={}, beta={}, topk={}, normalize_similarity={})'.format(
            self.alpha, self.beta, self.topK, self.normalize_similarity)

    def fit(self, alpha=0.7262744966754912, beta=0.583348423295889, topK=100, normalize_similarity=True):
        self.alpha = alpha
        self.beta = beta
        self.topK = topK

        self.normalize_similarity = normalize_similarity

        #Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l2', axis=1)

        #Piu is the column-normalized, 'boolean' urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        degree = np.zeros(self.URM_train.shape[1])
        nonZeroMask = X_bool_sum!=0.0
        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)

        #ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l2', axis=1)
        del(X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):
            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = np.multiply(similarity_block[row_in_block, :], degree)
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

            if time.time() - start_time_printBatch > 300:
                new_time_value, new_time_unit = seconds_to_biggest_unit(time.time() - start_time)

                self._print('Similarity column {} ({:4.1f}%), {:.2f} column/sec. Elapsed time {:.2f} {}'.format(
                     current_block_start_row + block_dim,
                    100.0 * float(current_block_start_row + block_dim) / Pui.shape[1],
                    float(current_block_start_row + block_dim) / (time.time() - start_time),
                    new_time_value, new_time_unit))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l2', axis=1)

        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)

        self.W_sparse = check_matrix(self.W_sparse, format='csr')