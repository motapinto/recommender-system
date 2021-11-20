import os
from Utils.Dataset import Dataset
from Recommenders.CF.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.CF.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.CF.KNN.RP3betaRecommender import RP3betaRecommender
from Recommenders.CF.KNN.P3alphaRecommender import P3alphaRecommender
from Recommenders.CF.KNN.EASE_R_Recommender import EASE_R_Recommender
from Utils.submission import get_submission, save_submission

if __name__ == '__main__':
    dataset = Dataset(path='./Data', train_percentage=0.9, val_split=False)
    recommender = UserKNNCFRecommender(dataset.URM_train)
    
    recommender.fit()
    # output_folder_path = os.path.join('Recommenders', 'tuner_results'+os.sep)
    # recommender.load_model(
    #     output_folder_path,
    #     file_name=recommender.RECOMMENDER_NAME+'_best_model_last.zip'
    # )

    submission = get_submission(dataset.targets, recommender)
    save_submission(submission)