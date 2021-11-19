import os
from Utils.Dataset import Dataset
from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python
from Utils.submission import get_submission, save_submission

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation=0.1, testing=0.1)

    recommender = SLIM_BPR_Python(dataset.URM_train)
    recommender.fit()

    submission = get_submission(dataset.targets, recommender)
    save_submission(submission)