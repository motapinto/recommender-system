import os
import time
import numpy as np
from Utils.DataIO import DataIO
from Recommenders.recommender_utils import check_matrix
from Utils.Evaluator import EvaluatorHoldout

class Base(object):
    RECOMMENDER_NAME = 'Recommender_Base_Class'

    def __init__(self, URM_train, verbose=True):
        super(Base, self).__init__()

        self.URM_train = check_matrix(URM_train.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()

        self.n_users, self.n_items = self.URM_train.shape
        self.verbose = verbose

        self.filterTopPop_ItemsID = np.array([], dtype=np.int)

        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)

        self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0

        if self._cold_user_mask.any():
            self._print('URM Detected {} ({:4.1f}%) users with no interactions.'.format(
                self._cold_user_mask.sum(), self._cold_user_mask.sum()/self.n_users*100))

        self._cold_item_mask = np.ediff1d(self.URM_train.tocsc().indptr) == 0

        if self._cold_item_mask.any():
            self._print('URM Detected {} ({:4.1f}%) items with no interactions.'.format(
                self._cold_item_mask.sum(), self._cold_item_mask.sum()/self.n_items*100))

    def _get_cold_user_mask(self): return self._cold_user_mask

    def _get_cold_item_mask(self): return self._cold_item_mask

    def _print(self, string):
        if self.verbose: print('{}: {}'.format(self.RECOMMENDER_NAME, string))

    def fit(self): pass

    def get_URM_train(self): return self.URM_train.copy()

    def set_URM_train(self, URM_train_new, **kwargs):
        assert self.URM_train.shape == URM_train_new.shape, '{}: set_URM_train old and new URM train have different shapes'.format(self.RECOMMENDER_NAME)

        if len(kwargs)>0:
            self._print('set_URM_train keyword arguments not supported for this recommender class. Received: {}'.format(kwargs))

        self.URM_train = check_matrix(URM_train_new.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()

        self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0

        if self._cold_user_mask.any():
            self._print('Detected {} ({:4.1f}%) users with no interactions.'.format(
                self._cold_user_mask.sum(), self._cold_user_mask.sum()/len(self._cold_user_mask)*100))

    def _remove_seen_on_scores(self, user_id, scores):
        assert self.URM_train.getformat() == 'csr', 'Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items'
        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
        scores[seen] = -np.inf
        return scores

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        raise NotImplementedError('Base: compute_item_score not assigned for current recommender, unable to compute prediction scores')

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None, return_scores=False):
        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None: cutoff = self.URM_train.shape[1] - 1
        cutoff = min(cutoff, self.URM_train.shape[1] - 1)

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)

        for user_index in range(len(user_id_array)):
            user_id = user_id_array[user_index]
            if remove_seen_flag:
                scores_batch[user_index,:] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:,0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = [None] * ranking.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(len(user_id_array)):
            user_recommendation_list = ranking[user_index]
            user_item_scores = scores_batch[user_index, user_recommendation_list]

            not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

            user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
            ranking_list[user_index] = user_recommendation_list.tolist()

        # Return single list for one user, instead of list of lists
        if single_user: ranking_list = ranking_list[0]

        if return_scores: return ranking_list, scores_batch
        else: return ranking_list

    def save_model(self, folder_path, file_name=None):
        raise NotImplementedError('Base: save_model not implemented')

    def load_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print(f'Loading model from file {folder_path + file_name}')

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])

        self._print('Loading complete')

    def evaluate_model(self, URM_test, fit_params={}, load=True):
        start = time.time()
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

        if len(fit_params) == 0 and load:
            output_folder_path = os.path.join('Recommenders', 'saved_models', 'test'+os.sep)
            self.load_model(output_folder_path)
        else:
            self.fit(**fit_params)

        result_df, _ = evaluator_test.evaluateRecommender(self)
        end = time.time()

        map = result_df.loc[10]['MAP']
        exec_time = int(end-start)
        print(f'\nRecommender performance: MAP = {map}. Time: {exec_time} s.\n')   

        return result_df, exec_time
