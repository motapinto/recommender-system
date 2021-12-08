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

def evaluate_recommender(recommender_class, URM_train, ICM, URM_test):
    if recommender_class == ItemKNN_CFCBF_Hybrid:
        recommender = recommender_class(URM_train, ICM)
    else:
        recommender = recommender_class(URM_train)
    
    recommender.fit()

    start = time.time()
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_df, _ = evaluator_test.evaluateRecommender(recommender)
    end = time.time()

    return result_df, end-start

if __name__ == '__main__':
    dataset = Dataset(path='./Data', k=5, validation_percentage=0, test_percentage=0.2)
    test_models = [ItemKNN_CFCBF_Hybrid ]#, UserKNNCF, ItemKNNCF, EASE_R, RP3beta]

    if dataset.cross_val:
        avg_results = []
        for model in test_models:
            result_array = []
            for i in trange(dataset.k):                
                result_df, exec_time = evaluate_recommender(model, 
                    dataset.URM_trains[i], dataset.ICM, dataset.URM_tests[i])
                
                result_array.append(result_df.loc[10]['MAP'])
                print('\nRecommender performance: MAP = {:.4f}. Time: {} s.\n'.format(
                    result_df.loc[10]['MAP'], exec_time))
            
            avg_results.append(average(result_array))

        for model_idx, map in enumerate(avg_results):
            print(f'Recommender: {test_models[model_idx].RECOMMENDER_NAME} | MAP@10: {map}')
            
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


        