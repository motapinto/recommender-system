import os
from Utils.Dataset import Dataset
from Utils.methods.get_recommender_class import get_recommender_class
from Utils.submission import get_submission, save_submission

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation=0.1, testing=0.1)

    cf_models = ['ItemKNNCFRecommender', 'UserKNNCFRecommender']
    model_name = cf_models[1]

    recommender_class = get_recommender_class(model_name)
    recommender = recommender_class(dataset.URM_train)
    recommender.fit()

    submission = get_submission(dataset.targets, recommender)
    save_submission(submission)