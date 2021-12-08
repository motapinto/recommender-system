import time
from tqdm import trange

from numpy.lib.function_base import average
from Recommenders.CF.MatrixFactorization.IALS import IALS
from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout

from Recommenders.CF.KNN.ItemKNNCF import ItemKNNCF
from Recommenders.CF.KNN.UserKNNCF import UserKNNCF
from Recommenders.CF.KNN.RP3beta import RP3beta
from Recommenders.CF.KNN.P3alpha import P3alpha
from Recommenders.CF.KNN.EASE_R import EASE_R
from Recommenders.CF.KNN.MachineLearning.SLIM_BPR import SLIM_BPR
from Recommenders.CF.KNN.MachineLearning.SLIMElasticNet import SLIMElasticNet
from Recommenders.CF.MatrixFactorization.PureSVD import PureSVD, ScaledPureSVD
from Recommenders.CF.MatrixFactorization.PureSVDItem import PureSVDItem

from Recommenders.Hybrid.ItemKNN_CFCBF_Hybrid import ItemKNN_CFCBF_Hybrid

def evaluate_recommender(recommender, URM_train, ICM, URM_test, has_fit_params=False):
    start = time.time()

    if not has_fit_params:
        if recommender == ItemKNN_CFCBF_Hybrid:
            recommender = recommender(URM_train, ICM)
        else:
            recommender = recommender(URM_train)
        
        recommender.fit()

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_df, _ = evaluator_test.evaluateRecommender(recommender)
    end = time.time()

    return result_df, int(end-start)

def evaluate_models(k, URM_trains, ICM, URM_tests, test_models):
    avg_results = []
    for model in test_models:
        result_array = []
        for i in trange(k):                
            result_df, exec_time = evaluate_recommender(model, 
                URM_trains[i], ICM, URM_tests[i])
            
            result_array.append(result_df.loc[10]['MAP'])
            print('\nRecommender performance: MAP = {:.4f}. Time: {} s.\n'.format(
                result_df.loc[10]['MAP'], exec_time))
        
        avg_results.append(average(result_array))

    for model_idx, map in enumerate(avg_results):
        print(f'Recommender: {test_models[model_idx].RECOMMENDER_NAME} | MAP@10: {map}')

if __name__ == '__main__':
    dataset = Dataset(path='./Data', k=0, validation_percentage=0, test_percentage=0.2)
    
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
        ICM = dataset.get_icm_format_k(11)
        stacked_URM, stacked_ICM = dataset.stack_URM_ICM(dataset.URM_train, ICM)

        result_df, exec_time = evaluate_recommender(ItemKNNCF, 
            stacked_URM, stacked_ICM, dataset.URM_test)
        map = result_df.loc[10]['MAP']
        print(f'\nRecommender performance: MAP = {map}. Time: {exec_time} s.\n')

        ICM = dataset.get_icm_format_k(1)

        result_df, exec_time = evaluate_recommender(ItemKNNCF, 
            dataset.URM_train, ICM, dataset.URM_test)
        map = result_df.loc[10]['MAP']
        print(f'\nRecommender performance: MAP = {map}. Time: {exec_time}s.\n')

            
    # if recommender_class == IALS or recommender_class == SLIM_BPR or recommender_class == SLIMElasticNet:
    #     earlystopping_keywargs = {
    #         'validation_every_n': 5,
    #         'stop_on_validation': True,
    #         'evaluator_object': EvaluatorHoldout(dataset.URM_test, cutoff_list=[10]),
    #         'lower_validations_allowed': 5,
    #         'validation_metric': 'MAP',
    #     }   
    #     recommender.fit(**earlystopping_keywargs)
    # else: 
    # recommender.fit()


        
# ItemKNN_CFCBF_Hybrid - MAP: 0.2345
# UserKNNCF - MAP: 0.2318
# ItemKNNCF - MAP: 0.2398
# EASE_R - MAP: 0.2457
# RP3beta - MAP: 0.2204