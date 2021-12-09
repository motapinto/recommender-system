import time
from tqdm import trange

from numpy.lib.function_base import average
from Recommenders.CF.MatrixFactorization.IALS import IALS
from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Utils.methods.get_recommender_instance import get_recommender_instance

from Recommenders.CF.KNN.ItemKNNCF import ItemKNNCF
from Recommenders.CF.KNN.UserKNNCF import UserKNNCF
from Recommenders.CF.KNN.RP3beta import RP3beta
from Recommenders.CF.KNN.P3alpha import P3alpha
from Recommenders.CF.KNN.EASE_R import EASE_R
from Recommenders.CF.KNN.SLIM_BPR import SLIM_BPR
from Recommenders.CF.KNN.SLIMElasticNet import SLIMElasticNet
from Recommenders.CF.MatrixFactorization.PureSVD import PureSVD, ScaledPureSVD
from Recommenders.CF.MatrixFactorization.PureSVDItem import PureSVDItem

from Recommenders.Hybrid.ItemKNN_CFCBF_Hybrid import ItemKNN_CFCBF_Hybrid

def evaluate_recommender(recommender_class, URM_train, ICM, URM_test, fit_params={}):
    start = time.time()
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    recommender = get_recommender_instance(recommender_class, URM_train, ICM)
    recommender.fit(**fit_params)

    result_df, _ = evaluator_test.evaluateRecommender(recommender)
    end = time.time()

    return result_df, int(end-start)

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
        params = {
            'num_factors': 24,
            'topK': 853, 
        }

        result_df, exec_time = evaluate_recommender(PureSVDItem, 
            stacked_URM, stacked_ICM, dataset.URM_test, fit_params=params)
        map = result_df.loc[10]['MAP']
        print(f'\nRecommender performance: MAP = {map}. Time: {exec_time} s.\n')

        result_df, exec_time = evaluate_recommender(PureSVDItem, 
            stacked_URM, stacked_ICM, dataset.URM_test)
        map = result_df.loc[10]['MAP']
        print(f'\nRecommender performance: MAP = {map}. Time: {exec_time} s.\n')

        
# ItemKNN_CFCBF_Hybrid - MAP: 0.2345
# UserKNNCF - MAP: 0.2318
# ItemKNNCF - MAP: 0.2398
# EASE_R - MAP: 0.2457
# RP3beta - MAP: 0.2204
# PureSVDItem - MAP: 0.2271