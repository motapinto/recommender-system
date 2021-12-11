import os, sys
from Recommenders.recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.Base.BaseItemSimilarityMatrix import BaseItemSimilarityMatrix
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.CF.KNN.Cython.SLIM_BPR_Epoch import SLIM_BPR_Epoch

class SLIM_BPR(BaseItemSimilarityMatrix, Incremental_Training_Early_Stopping):
    RECOMMENDER_NAME = 'SLIM_BPR'

    def __init__(self, URM_train, verbose=True):
        super(SLIM_BPR, self).__init__(URM_train, verbose=verbose)

    def fit(
        self, epochs=300,
        symmetric = True,
        random_seed = None,
        lambda_i = 0.0, lambda_j = 0.0, learning_rate = 1e-4, topK = 200,
        sgd_mode='adagrad', gamma=0.995, beta_1=0.9, beta_2=0.999,
        **earlystopping_kwargs
    ):
        self.symmetric = symmetric
        self.train_with_sparse_weights=True

        # Select only positive interactions
        URM_train_positive = self.URM_train.copy()

        self.sgd_mode = sgd_mode
        self.epochs = epochs

        self.cythonEpoch = SLIM_BPR_Epoch(
            URM_train_positive,
            train_with_sparse_weights=self.train_with_sparse_weights,
            final_model_sparse_weights=True,
            topK=topK,
            learning_rate=learning_rate,
            li_reg=lambda_i,
            lj_reg=lambda_j,
            symmetric=self.symmetric,
            sgd_mode=sgd_mode,
            verbose=self.verbose,
            random_seed=random_seed,
            gamma=gamma,
            beta_1=beta_1,
            beta_2=beta_2)

        if(topK != False and topK<1):
            raise ValueError('TopK not valid. Acceptable values are either False or a positive integer value. Provided value was \'{}\''.format(topK))
        self.topK = topK

        # self.batch_size = batch_size
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate

        self.S_incremental = self.cythonEpoch.get_S()
        self.S_best = self.S_incremental.copy()

        self._train_with_early_stopping(
            epochs,
            algorithm_name = self.RECOMMENDER_NAME,
            **earlystopping_kwargs)

        self.get_S_incremental_and_set_W()
        self.cythonEpoch._dealloc()
        sys.stdout.flush()

    def _prepare_model_for_validation(self):
        self.get_S_incremental_and_set_W()

    def _update_best_model(self):
        self.S_best = self.S_incremental.copy()

    def _run_epoch(self, num_epoch):
       self.cythonEpoch.epochIteration_Cython()

    def get_S_incremental_and_set_W(self):
        self.S_incremental = self.cythonEpoch.get_S()

        if self.train_with_sparse_weights:
            self.W_sparse = self.S_incremental
            self.W_sparse = check_matrix(self.W_sparse, format='csr')
        else:
            self.W_sparse = similarityMatrixTopK(self.S_incremental, k = self.topK)
            self.W_sparse = check_matrix(self.W_sparse, format='csr')
