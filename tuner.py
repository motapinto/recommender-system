import os
import traceback
from functools import partial
from skopt.space import Integer, Categorical
from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Recommenders.Search.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.CF.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.CF.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.CF.KNN.RP3betaRecommender import RP3betaRecommender
from Recommenders.CF.KNN.P3alphaRecommender import P3alphaRecommender
from Recommenders.CF.KNN.EASE_R_Recommender import EASE_R_Recommender

if __name__ == '__main__':
    dataset = Dataset(path='./Data', train_percentage=0.8, val_split=True)

    evaluator_validation = EvaluatorHoldout(dataset.URM_val, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(dataset.URM_test, cutoff_list=[10])

    output_folder_path = os.path.join('Recommenders', 'tuner_results'+os.sep)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    n_cases = 30
    run_hyperparameter_search_cf = partial(runHyperparameterSearch_Collaborative,
       URM_train=dataset.URM_train,
       URM_train_last_test=dataset.URM_train_val,
       metric_to_optimize='MAP',
       cutoff_to_optimize=10,
       evaluator_validation_earlystopping=evaluator_validation,
       evaluator_validation=evaluator_validation,
       evaluator_test=evaluator_test,
       output_folder_path=output_folder_path,
       parallelizeKNN=True,
       allow_weighting=True,
       resume_from_saved=False,
       save_model='best',
       similarity_type_list=['cosine', 'jaccard', 'asymmetric', 'dice', 'tversky'],
       n_cases=n_cases,
       n_random_starts=int(n_cases*0.3))
    
    cf_models = [
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        RP3betaRecommender,
        P3alphaRecommender,
        # EASE_R_Recommender -> fit() got an unexpected keyword argument 'validation_every_n'
    ]

    for recommender_class in cf_models:
        try: run_hyperparameter_search_cf(recommender_class)
        except Exception as e:
            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()
