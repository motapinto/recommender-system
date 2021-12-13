import time
from tqdm import trange
from numpy.lib.function_base import average

from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Utils.methods.get_recommender_instance import get_recommender_instance
from Utils.import_recommenders import *

def evaluate_recommender(recommender_class, URM_train, ICM, URM_test, fit_params={}):
    start = time.time()
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    recommender = get_recommender_instance(recommender_class, URM_train, ICM)
    recommender.fit(**fit_params)

    result_df, _ = evaluator_test.evaluateRecommender(recommender)
    end = time.time()

    map = result_df.loc[10]['MAP']
    exec_time = int(end-start)
    print(f'\nRecommender performance: MAP = {map}. Time: {exec_time} s.\n')   

    return result_df, exec_time

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
    dataset = Dataset(path='./Data', k=0, validation_percentage=0, test_percentage=0.2, seed=None)

    if dataset.cross_val:
        test_models = [
            ItemKNN_CFCBF_Hybrid,
            UserKNNCF,
            ItemKNNCF,
            EASE_R,
            RP3beta
        ]

        evaluate_models(dataset.k, dataset.URM_trains, dataset.ICM, dataset.URM_tests, test_models)

    else:
        stacked_URM, stacked_ICM = dataset.stack_URM_ICM(dataset.URM_train, dataset.ICM)  

        evaluate_recommender(IALS, stacked_URM.copy(), 
            stacked_ICM.copy(), dataset.URM_test)

        fit_params = {'num_factors': 32, 'epochs': 30, 'confidence_scaling': 'linear', 'alpha': 0.41481077075270684, 'epsilon': 0.008058602991460452, 'reg': 1.1458744991769949e-05}
        evaluate_recommender(IALS, stacked_URM.copy(), 
            stacked_ICM.copy(), dataset.URM_test, fit_params=fit_params)