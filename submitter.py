import os
from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Utils.submission import get_submission, save_submission
from Utils.import_recommenders import *

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation_percentage=0, test_percentage=0, seed=None)
    ICM = dataset.get_icm_format_k(11)
    stacked_URM, _ = dataset.stack_URM_ICM(dataset.URM_train, ICM)
    
    recommender = MultiThreadSLIM_SLIMElasticNet(stacked_URM)
    recommender.fit()

    submission = get_submission(dataset.targets, recommender)
    save_submission(submission)