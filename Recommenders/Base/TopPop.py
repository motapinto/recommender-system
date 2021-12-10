import numpy as np
from Recommenders.Base.Base import Base
from Recommenders.recommender_utils import check_matrix
from Utils.DataIO import DataIO

class TopPop(Base):
    '''Top Popular recommender'''

    RECOMMENDER_NAME = 'TopPopRecommender'

    def __init__(self, URM_train):
        super(TopPop, self).__init__(URM_train)

    def fit(self):
        self.item_pop = np.ediff1d(self.URM_train.tocsc().indptr)
        self.n_items = self.URM_train.shape[1]

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        if items_to_compute is not None:
            item_pop_to_copy = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_pop_to_copy[items_to_compute] = self.item_pop[items_to_compute].copy()
        else:
            item_pop_to_copy = self.item_pop.copy()

        item_scores = np.array(item_pop_to_copy, dtype=np.float32).reshape((1, -1))
        item_scores = np.repeat(item_scores, len(user_id_array), axis = 0)

        return item_scores

    def save_model(self, folder_path, file_name = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"item_pop": self.item_pop}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")