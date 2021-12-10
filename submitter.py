import os
from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Utils.submission import get_submission, save_submission

from Recommenders.CF.KNN.ItemKNNCF import ItemKNNCF
from Recommenders.CF.KNN.UserKNNCF import UserKNNCF
from Recommenders.CF.KNN.RP3beta import RP3beta
from Recommenders.CF.KNN.P3alpha import P3alpha
from Recommenders.CF.KNN.EASE_R import EASE_R
from Recommenders.CF.KNN.SLIM_BPR import SLIM_BPR
from Recommenders.CF.KNN.SLIMElasticNet import SLIMElasticNet
from Recommenders.CF.MatrixFactorization.PureSVD import PureSVD, ScaledPureSVD
from Recommenders.CF.MatrixFactorization.PureSVDItem import PureSVDItem
from Recommenders.CF.MatrixFactorization.IALS import IALS

from Recommenders.Hybrid.ItemKNN_CFCBF_Hybrid import ItemKNN_CFCBF_Hybrid
from Recommenders.Hybrid.Hybrid1 import Hybrid1

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation_percentage=0, test_percentage=0, seed=None)
    ICM = dataset.get_icm_format_k(11)
    stacked_URM, _ = dataset.stack_URM_ICM(dataset.URM_train, ICM)
    
    recommender = Hybrid1(stacked_URM, dataset.ICM)
    recommender.fit()

    submission = get_submission(dataset.targets, recommender)
    save_submission(submission)