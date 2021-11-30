import numpy as np
import scipy.sparse as sps

def compute_W_sparse_from_item_latent_factors(ITEM_factors, topK = 100):
    n_items, _ = ITEM_factors.shape
    block_size = 100
    start_item = 0
    end_item = 0

    values = []
    rows = []
    cols = []

    # Compute all similarities for each item using vectorization
    while start_item < n_items:
        end_item = min(n_items, start_item + block_size)
        this_block_weight = np.dot(ITEM_factors[start_item:end_item, :], ITEM_factors.T)

        for col_index_in_block in range(this_block_weight.shape[0]):
            this_column_weights = this_block_weight[col_index_in_block, :]
            item_original_index = start_item + col_index_in_block

            relevant_items_partition = (-this_column_weights).argpartition(topK-1)[0:topK]
            relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
            top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

            # Incrementally build sparse matrix, do not add zeros
            notZerosMask = this_column_weights[top_k_idx] != 0.0
            numNotZeros = np.sum(notZerosMask)

            values.extend(this_column_weights[top_k_idx][notZerosMask])
            rows.extend(top_k_idx[notZerosMask])
            cols.extend(np.ones(numNotZeros) * item_original_index)

        start_item += block_size

    W_sparse = sps.csr_matrix(
        (values, (rows, cols)),
        shape=(n_items, n_items),
        dtype=np.float32)

    return W_sparse