import os
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

from Recommenders.Hybrid.Hybrid import Hybrid

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation_percentage=0, test_percentage=0.1)
    ICM_normalized, aggregated_matrixes_1, aggregated_matrixes_2 = dataset.aggregate_matrixes()

    evaluator_test = EvaluatorHoldout(dataset.URM_test, cutoff_list=[10])
    recommender = Hybrid(dataset.URM_train, dataset.ICM['genre_ICM'])
    recommender.fit()

    result_df, _ = evaluator_test.evaluateRecommender(recommender)
    print(result_df.loc[10])

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

    # result_df, _ = evaluator_test.evaluateRecommender(recommender)
    # precision_metric = result_df.loc[10]['Precision']
    # map_metric = result_df.loc[10]['MAP']

    # output_folder_path = os.path.join('Recommenders', 'tuner_results'+os.sep)
    # filename = recommender.RECOMMENDER_NAME+'_best_model_last.zip'
    # # if os.path.isfile(os.path.join(output_folder_path, filename)):
    # #     best_recommender = get_recommender_instance(recommender_class, dataset.URM_train, dataset.ICM)
    # #     best_recommender.load_model(output_folder_path, filename)

    # #     best_result_df, _ = evaluator_test.evaluateRecommender(best_recommender)
    # #     best_precision_metric = result_df.loc[10]['Precision']
    # #     best_map_metric = result_df.loc[10]['MAP']

    # #     if best_map_metric < map_metric && best_precision_metric < map_metric:
    # #         best_result_df = result_df
    # #         recommender.save_model(output_folder_path, filename)
    # # else: 
        
    # # recommender.save_model(output_folder_path, filename)
    # print(result_df.loc[10])
        