import os
from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Utils.submission import get_submission, save_submission

from Recommenders.CF.KNN.ItemKNNCF import ItemKNNCF
from Recommenders.CF.KNN.UserKNNCF import UserKNNCF
from Recommenders.CF.KNN.RP3beta import RP3beta
from Recommenders.CF.KNN.P3alpha import P3alpha
from Recommenders.CF.KNN.EASE_R import EASE_R
from Recommenders.CF.KNN.MachineLearning.SLIM_BPR import SLIM_BPR
from Recommenders.CF.KNN.MachineLearning.SLIMElasticNet import SLIMElasticNet
from Recommenders.CF.MatrixFactorization.PureSVD import PureSVD, ScaledPureSVD
from Recommenders.CF.MatrixFactorization.PureSVDItem import PureSVDItem
from Recommenders.CF.MatrixFactorization.IALS import IALS

from Recommenders.Hybrid.Hybrid import Hybrid

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation_percentage=0, test_percentage=0.1)

    recommender = Hybrid(dataset.URM_train, dataset.ICM['genre_ICM'])
    recommender.fit()

    # recommender.load_model(
    #     os.path.join('Recommenders', 'tuner_results'+os.sep),
    #     file_name=recommender.RECOMMENDER_NAME+'_best_model_last.zip'
    # )

    submission = get_submission(dataset.targets, recommender)
    save_submission(submission)