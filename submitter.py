import os
import numpy as np
from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Utils.submission import get_submission, save_submission
from Utils.import_recommenders import *

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation_percentage=0, test_percentage=0, seed=None)
    stacked_URM, _ = dataset.stack_URM_ICM(dataset.URM_train, dataset.ICM)
    
    recommender = Hybrid(stacked_URM, dataset.ICM, 'prod')
    recommender.fit()

    submission = get_submission(dataset.targets, recommender)
    save_submission(submission)