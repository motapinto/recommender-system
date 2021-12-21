import os
import numpy as np
from tqdm import trange
from numpy.lib.function_base import average

from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Utils.methods.get_recommender_instance import get_recommender_instance
from Utils.import_recommenders import *
from Utils.methods.ir_feature_weighting import TF_IDF, okapi_BM_25

def evaluate_models(k, URM_trains, ICM, URM_tests, test_models):
    avg_results = []
    for model in test_models:
        result_array = []

        for i in trange(k):     
            stacked_URM, _ = dataset.stack_URM_ICM(URM_trains[i], ICM)           
            result_df, exec_time = evaluate_recommender(model, 
                stacked_URM, ICM, URM_tests[i])
            
            result_array.append(result_df.loc[10]['MAP'])
            print('\nRecommender performance: MAP = {:.4f}. Time: {} s.\n'.format(
                result_df.loc[10]['MAP'], exec_time))
        
        avg_results.append(average(result_array))

    for model_idx, map in enumerate(avg_results):
        print(f'Recommender: {test_models[model_idx].RECOMMENDER_NAME} | MAP@10: {map}')

if __name__ == '__main__':
    dataset = Dataset(path='./Data', k=0, validation_percentage=0, test_percentage=0.2, seed=1234)

    if dataset.cross_val:
        test_models = [
            #ItemKNN_CFCBF_Hybrid,
            #UserKNNCF,
            ItemKNNCF,
            #EASE_R,
            #RP3beta
        ]

        evaluate_models(dataset.k, dataset.URM_trains, dataset.ICM, dataset.URM_tests, test_models)

    else:
        stacked_URM, stacked_ICM = dataset.stack_URM_ICM(dataset.URM_train.copy(), dataset.ICM.copy())  
        output_folder_path = os.path.join('Recommenders', 'saved_models', 'test'+os.sep)

        # model = ItemKNNCF(stacked_URM.copy()) #  MAP = 0.2413
        # model.evaluate_model(dataset.URM_test)

        # fit_params = {'topK': 100, 'shrink': 298, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.056659072281978876, 'feature_weighting': 'TF-IDF', 'URM_bias': 0.0963202309810741}
        # model = ItemKNNCF(stacked_URM.copy())
        # model.evaluate_model(dataset.URM_test, fit_params)
        model = Hybrid4(stacked_URM.copy(), stacked_ICM.copy())
        model.evaluate_model(dataset.URM_test) # 0.24926



