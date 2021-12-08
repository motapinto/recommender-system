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

from Recommenders.CF.KNN.EASE_R import EASE_R

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation_percentage=0, test_percentage=0, seed=None)
    
    recommender = EASE_R(dataset.URM_train)
    recommender.fit()

    submission = get_submission(dataset.targets, recommender)
    save_submission(submission)