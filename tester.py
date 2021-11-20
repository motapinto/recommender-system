import os
from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Recommenders.CF.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.CF.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.CF.KNN.RP3betaRecommender import RP3betaRecommender
from Recommenders.CF.KNN.P3alphaRecommender import P3alphaRecommender
from Recommenders.CF.KNN.EASE_R_Recommender import EASE_R_Recommender

if __name__ == '__main__':
    dataset = Dataset(path='./Data', train_percentage=0.9, val_split=False)
    evaluator_test = EvaluatorHoldout(dataset.URM_test, cutoff_list=[10])
    recommender = UserKNNCFRecommender(dataset.URM_train)
    
    output_folder_path = os.path.join('Recommenders', 'tuner_results'+os.sep)
    
    try:
        recommender.load_model(
            output_folder_path,
            file_name=recommender.RECOMMENDER_NAME+'_best_model_last.zip'
        )
    except Exception as e:
        recommender.fit()
    finally:
        _, results_run_string = evaluator_test.evaluateRecommender(recommender)
        print(results_run_string)
        