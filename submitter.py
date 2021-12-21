import os
import numpy as np
from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Utils.submission import get_submission, save_submission
from Utils.import_recommenders import *

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation_percentage=0, test_percentage=0, seed=None)
    stacked_URM, _ = dataset.stack_URM_ICM(dataset.URM_train, dataset.ICM)
    
    recommender = Hybrid4(stacked_URM, dataset.ICM, 'prod')
    recommender.fit()

    # {'alpha': 0.2332175235180347, 'l1_ratio': 9.480795250283082e-05, 'topK': 1000} submit_3
    # {'alpha': 0.006727591135116065, 'l1_ratio': 0.022041069606479908, 'topK': 480} submit_4

    submission = get_submission(dataset.targets, recommender)
    save_submission(submission)