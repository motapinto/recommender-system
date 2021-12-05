import os, multiprocessing
from functools import partial
from skopt.space import Real, Integer, Categorical
import traceback

from Recommenders.Search.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.Search.SearchAbstractClass import SearchInputRecommenderArgs

# KNN
from Recommenders.CF.KNN.ItemKNNCF import ItemKNNCF
from Recommenders.CF.KNN.UserKNNCF import UserKNNCF
from Recommenders.CF.KNN.P3alpha import P3alpha
from Recommenders.CF.KNN.RP3beta import RP3beta
from Recommenders.CF.KNN.EASE_R import EASE_R

# KNN machine learning
from Recommenders.CF.KNN.MachineLearning.SLIM_BPR import SLIM_BPR
from Recommenders.CF.KNN.MachineLearning.SLIMElasticNet import SLIMElasticNet, MultiThreadSLIM_SLIMElasticNet

# Matrix Factorization
from Recommenders.CF.MatrixFactorization.PureSVD import PureSVD
from Recommenders.CF.MatrixFactorization.PureSVDItem import PureSVDItem
from Recommenders.CF.MatrixFactorization.IALS import IALS
from Recommenders.CF.MatrixFactorization.LightFM import LightFMCF
# from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
# from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython,\
# MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
# from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask as MultVAERecommender

# from Recommenders.KNN.ItemKNNCBF import ItemKNNCBF
# from Recommenders.KNN.UserKNNCBF import UserKNNCBF

# from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
# from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
# from Recommenders.FactorizationMachines.LightFM import LightFMItemHybridRecommender, LightFMUserHybridRecommender
# from Recommenders.FeatureWeighting.Cython.CFW_D_Similarity_Cython import CFW_D_Similarity_Cython
# from Recommenders.FeatureWeighting.Cython.CFW_DVV_Similarity_Cython import CFW_DVV_Similarity_Cython
# from Recommenders.FeatureWeighting.Cython.FBSM_Rating_Cython import FBSM_Rating_Cython

def runHyperparameterSearch_FeatureWeighting(
    recommender_class, URM_train, W_train, ICM_object, ICM_name, n_cases = None,
    evaluator_validation= None, evaluator_test=None, max_total_time = None,
    metric_to_optimize = None, cutoff_to_optimize = None,
    evaluator_validation_earlystopping = None, resume_from_saved = False, save_model = 'best',
    output_folder_path ='result_experiments/',
    similarity_type_list = None
):
    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_file_name_root = recommender_class.RECOMMENDER_NAME + '_{}'.format(ICM_name)
    hyperparameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

    # if recommender_class is FBSM_Rating_Cython:
    #     hyperparameters_range_dictionary = {
    #         'topK': Categorical([300]),
    #         'n_factors': Integer(1, 5),
    #         'learning_rate': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
    #         'sgd_mode': Categorical(['adam']),
    #         'l2_reg_D': Real(low = 1e-6, high = 1e1, prior = 'log-uniform'),
    #         'l2_reg_V': Real(low = 1e-6, high = 1e1, prior = 'log-uniform'),
    #         'epochs': Categorical([300]),
    #     }

    #     recommender_input_args = SearchInputRecommenderArgs(
    #         CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_object],
    #         CONSTRUCTOR_KEYWORD_ARGS = {},
    #         FIT_POSITIONAL_ARGS = [],
    #         FIT_KEYWORD_ARGS = {
    #             'validation_every_n': 5,
    #             'stop_on_validation': True,
    #             'evaluator_object': evaluator_validation_earlystopping,
    #             'lower_validations_allowed': 10,
    #             'validation_metric': metric_to_optimize
    #         }
    #     )

    # if recommender_class is CFW_D_Similarity_Cython:
    #     hyperparameters_range_dictionary = {
    #         'topK': Categorical([300]),

    #         'learning_rate': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
    #         'sgd_mode': Categorical(['adam']),
    #         'l1_reg': Real(low = 1e-3, high = 1e-2, prior = 'log-uniform'),
    #         'l2_reg': Real(low = 1e-3, high = 1e-1, prior = 'log-uniform'),
    #         'epochs': Categorical([50]),

    #         'init_type': Categorical(['one', 'random']),
    #         'add_zeros_quota': Real(low = 0.50, high = 1.0, prior = 'uniform'),
    #         'positive_only_weights': Categorical([True, False]),
    #         'normalize_similarity': Categorical([True]),

    #         'use_dropout': Categorical([True]),
    #         'dropout_perc': Real(low = 0.30, high = 0.8, prior = 'uniform'),
    #     }

    #     recommender_input_args = SearchInputRecommenderArgs(
    #         CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_object, W_train],
    #         CONSTRUCTOR_KEYWORD_ARGS = {},
    #         FIT_POSITIONAL_ARGS = [],
    #         FIT_KEYWORD_ARGS = {
    #             'precompute_common_features':False,     # Reduces memory requirements
    #             'validation_every_n': 5,
    #             'stop_on_validation': True,
    #             'evaluator_object': evaluator_validation_earlystopping,
    #             'lower_validations_allowed': 10,
    #             'validation_metric': metric_to_optimize
    #         }
    #     )

    # if recommender_class is CFW_DVV_Similarity_Cython:
    #     hyperparameters_range_dictionary = {
    #         'topK': Categorical([300]),
    #         'n_factors': Integer(1, 2),

    #         'learning_rate': Real(low = 1e-5, high = 1e-3, prior = 'log-uniform'),
    #         'sgd_mode': Categorical(['adam']),
    #         'l2_reg_D': Real(low = 1e-6, high = 1e1, prior = 'log-uniform'),
    #         'l2_reg_V': Real(low = 1e-6, high = 1e1, prior = 'log-uniform'),
    #         'epochs': Categorical([100]),

    #         'add_zeros_quota': Real(low = 0.50, high = 1.0, prior = 'uniform'),
    #     }

    #     recommender_input_args = SearchInputRecommenderArgs(
    #         CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_object, W_train],
    #         CONSTRUCTOR_KEYWORD_ARGS = {},
    #         FIT_POSITIONAL_ARGS = [],
    #         FIT_KEYWORD_ARGS = {
    #             'precompute_common_features':False,     # Reduces memory requirements
    #             'validation_every_n': 5,
    #             'stop_on_validation': True,
    #             'evaluator_object': evaluator_validation_earlystopping,
    #             'lower_validations_allowed': 10,
    #             'validation_metric': metric_to_optimize
    #         }
    #     )

    ## Final step, after the hyperparameter range has been defined for each type of algorithm
    # hyperparameterSearch.search(
    #     recommender_input_args,
    #     hyperparameter_search_space= hyperparameters_range_dictionary,
    #     n_cases = n_cases,
    #     resume_from_saved = resume_from_saved,
    #     save_model = save_model,
    #     max_total_time = max_total_time,
    #     output_folder_path = output_folder_path,
    #     output_file_name_root = output_file_name_root,
    #     metric_to_optimize = metric_to_optimize,
    #     cutoff_to_optimize = cutoff_to_optimize)


# def runHyperparameterSearch_Hybrid(
#     recommender_class, URM_train, ICM_object, ICM_name, URM_train_last_test = None,
#     n_cases = None, n_random_starts = None, resume_from_saved = False,
#     save_model = 'best', evaluate_on_test = 'best', max_total_time = None, evaluator_validation_earlystopping = None,
#     evaluator_validation= None, evaluator_test=None, metric_to_optimize = None, cutoff_to_optimize = None,
#     output_folder_path ='result_experiments/', parallelizeKNN = False, allow_weighting = True,
#     similarity_type_list = None):
#     '''
#     This function performs the hyperparameter optimization for a hybrid collaborative and content-based recommender

#     :param recommender_class:   Class of the recommender object to optimize, it must be a Base type
#     :param URM_train:           Sparse matrix containing the URM training data
#     :param ICM_object:          Sparse matrix containing the ICM training data
#     :param ICM_name:            String containing the name of the ICM, will be used for the name of the output files
#     :param URM_train_last_test: Sparse matrix containing the union of URM training and validation data to be used in the last evaluation
#     :param n_cases:             Number of hyperparameter sets to explore
#     :param n_random_starts:     Number of the initial random hyperparameter values to explore, usually set at 30% of n_cases
#     :param resume_from_saved:   Boolean value, if True the optimization is resumed from the saved files, if False a new one is done
#     :param save_model:          ['no', 'best', 'last'] which of the models to save, see HyperparameterTuning/SearchAbstractClass for details
#     :param evaluate_on_test:    ['all', 'best', 'last', 'no'] when to evaluate the model on the test data, see HyperparameterTuning/SearchAbstractClass for details
#     :param max_total_time:    [None or int] if set stops the hyperparameter optimization when the time in seconds for training and validation exceeds the threshold
#     :param evaluator_validation:    Evaluator object to be used for the validation of each hyperparameter set
#     :param evaluator_validation_earlystopping:   Evaluator object to be used for the earlystopping of ML algorithms, can be the same of evaluator_validation
#     :param evaluator_test:          Evaluator object to be used for the test results, the output will only be saved but not used
#     :param metric_to_optimize:  String with the name of the metric to be optimized as contained in the output of the evaluator objects
#     :param cutoff_to_optimize:  Integer with the recommendation list length to be optimized as contained in the output of the evaluator objects
#     :param output_folder_path:  Folder in which to save the output files
#     :param parallelizeKNN:      Boolean value, if True the various heuristics of the KNNs will be computed in parallel, if False sequentially
#     :param allow_weighting:     Boolean value, if True it enables the use of TF-IDF and BM25 to weight features, users and items in KNNs
#     :param similarity_type_list: List of strings with the similarity heuristics to be used for the KNNs
#     '''

#     # If directory does not exist, create
#     if not os.path.exists(output_folder_path):
#         os.makedirs(output_folder_path)

#     URM_train = URM_train.copy()
#     ICM_object = ICM_object.copy()

#     if URM_train_last_test is not None: URM_train_last_test = URM_train_last_test.copy()

#     earlystopping_keywargs = {
#         'validation_every_n': 5,
#         'stop_on_validation': True,
#         'evaluator_object': evaluator_validation_earlystopping,
#         'lower_validations_allowed': 5,
#         'validation_metric': metric_to_optimize,
#     }


#     try:
#         output_file_name_root = recommender_class.RECOMMENDER_NAME + '_{}'.format(ICM_name)
#         hyperparameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

#         if recommender_class in [ItemKNN_CFCBF_Hybrid_Recommender, UserKNN_CFCBF_Hybrid_Recommender]:
#             if similarity_type_list is None:
#                 similarity_type_list = ['cosine', 'jaccard', 'asymmetric', 'dice', 'tversky']

#             hyperparameters_range_dictionary = {}

#             if recommender_class is ItemKNN_CFCBF_Hybrid_Recommender:
#                 hyperparameters_range_dictionary['ICM_weight'] = Real(low = 1e-2, high = 1e2, prior = 'log-uniform')

#             elif recommender_class is UserKNN_CFCBF_Hybrid_Recommender:
#                 hyperparameters_range_dictionary['UCM_weight'] = Real(low = 1e-2, high = 1e2, prior = 'log-uniform')




#             recommender_input_args = SearchInputRecommenderArgs(
#                 CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_object],
#                 CONSTRUCTOR_KEYWORD_ARGS = {},
#                 FIT_POSITIONAL_ARGS = [],
#                 FIT_KEYWORD_ARGS = {}
#             )


#             if URM_train_last_test is not None:
#                 recommender_input_args_last_test = recommender_input_args.copy()
#                 recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
#             else:
#                 recommender_input_args_last_test = None


#             run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNRecommender_on_similarity_type,
#                                                            hyperparameter_search_space = hyperparameters_range_dictionary,
#                                                            recommender_input_args = recommender_input_args,
#                                                            hyperparameterSearch = hyperparameterSearch,
#                                                            resume_from_saved = resume_from_saved,
#                                                            save_model = save_model,
#                                                            evaluate_on_test = evaluate_on_test,
#                                                            max_total_time = max_total_time,
#                                                            n_cases = n_cases,
#                                                            n_random_starts = n_random_starts,
#                                                            output_folder_path = output_folder_path,
#                                                            output_file_name_root = output_file_name_root,
#                                                            metric_to_optimize = metric_to_optimize,
#                                                            cutoff_to_optimize = cutoff_to_optimize,
#                                                            allow_weighting = allow_weighting,
#                                                            recommender_input_args_last_test = recommender_input_args_last_test)



#             if parallelizeKNN:
#                 pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
#                 resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

#                 pool.close()
#                 pool.join()

#             else:

#                 for similarity_type in similarity_type_list:
#                     run_KNNCFRecommender_on_similarity_type_partial(similarity_type)


#             return


#         elif recommender_class in [LightFMItemHybridRecommender, LightFMUserHybridRecommender]:
#                 hyperparameters_range_dictionary = {
#                     'epochs': Categorical([300]),
#                     'n_components': Integer(1, 200),
#                     'loss': Categorical(['bpr', 'warp', 'warp-kos']),
#                     'sgd_mode': Categorical(['adagrad', 'adadelta']),
#                     'learning_rate': Real(low = 1e-6, high = 1e-1, prior = 'log-uniform'),
#                     'item_alpha': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
#                     'user_alpha': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
#                 }

#                 recommender_input_args = SearchInputRecommenderArgs(
#                     CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_object],
#                     CONSTRUCTOR_KEYWORD_ARGS = {},
#                     FIT_POSITIONAL_ARGS = [],
#                     FIT_KEYWORD_ARGS = earlystopping_keywargs
#                 )

#         if URM_train_last_test is not None:
#             recommender_input_args_last_test = recommender_input_args.copy()
#             recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
#         else:
#             recommender_input_args_last_test = None

#         ## Final step, after the hyperparameter range has been defined for each type of algorithm
#         hyperparameterSearch.search(
#             recommender_input_args,
#             hyperparameter_search_space= hyperparameters_range_dictionary,
#             n_cases = n_cases,
#             n_random_starts = n_random_starts,
#             resume_from_saved = resume_from_saved,
#             save_model = save_model,
#             evaluate_on_test = evaluate_on_test,
#             max_total_time = max_total_time,
#             output_folder_path = output_folder_path,
#             output_file_name_root = output_file_name_root,
#             metric_to_optimize = metric_to_optimize,
#             cutoff_to_optimize = cutoff_to_optimize,
#             recommender_input_args_last_test = recommender_input_args_last_test)

#     except Exception as e:
#         print('On recommender {} Exception {}'.format(recommender_class, str(e)))
#         traceback.print_exc()

#         error_file = open(output_folder_path + 'ErrorLog.txt', 'a')
#         error_file.write('On recommender {} Exception {}\n'.format(recommender_class, str(e)))
#         error_file.close()


def run_KNNRecommender_on_similarity_type(
    similarity_type, hyperparameterSearch,
    hyperparameter_search_space,
    recommender_input_args,
    n_cases,
    n_random_starts,
    resume_from_saved,
    save_model,
    evaluate_on_test,
    max_total_time,
    output_folder_path,
    output_file_name_root,
    metric_to_optimize,
    cutoff_to_optimize,
    allow_weighting = False,
    allow_bias_ICM = False,
    allow_bias_URM = False,
    recommender_input_args_last_test = None):

    original_hyperparameter_search_space = hyperparameter_search_space
    hyperparameters_range_dictionary = {
        'topK': Integer(5, 1000),
        'shrink': Integer(0, 1000),
        'similarity': Categorical([similarity_type]),
        'normalize': Categorical([True, False]),
    }

    is_set_similarity = similarity_type in ['tversky', 'dice', 'jaccard', 'tanimoto']

    if similarity_type == 'asymmetric':
        hyperparameters_range_dictionary['asymmetric_alpha'] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparameters_range_dictionary['normalize'] = Categorical([True])

    elif similarity_type == 'tversky':
        hyperparameters_range_dictionary['tversky_alpha'] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparameters_range_dictionary['tversky_beta'] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparameters_range_dictionary['normalize'] = Categorical([True])

    elif similarity_type == 'euclidean':
        hyperparameters_range_dictionary['normalize'] = Categorical([True, False])
        hyperparameters_range_dictionary['normalize_avg_row'] = Categorical([True, False])
        hyperparameters_range_dictionary['similarity_from_distance_mode'] = Categorical(['lin', 'log', 'exp'])

    if not is_set_similarity:
        if allow_weighting:
            hyperparameters_range_dictionary['feature_weighting'] = Categorical(['none', 'BM25', 'TF-IDF'])

        if allow_bias_ICM:
            hyperparameters_range_dictionary['ICM_bias'] = Real(low = 1e-2, high = 1e+3, prior = 'log-uniform')

        if allow_bias_URM:
            hyperparameters_range_dictionary['URM_bias'] = Real(low = 1e-2, high = 1e+3, prior = 'log-uniform')

    local_hyperparameter_search_space = {**hyperparameters_range_dictionary, **original_hyperparameter_search_space}

    hyperparameterSearch.search(
        recommender_input_args,
        hyperparameter_search_space= local_hyperparameter_search_space,
        n_cases = n_cases,
        n_random_starts = n_random_starts,
        resume_from_saved = resume_from_saved,
        save_model = save_model,
        evaluate_on_test = evaluate_on_test,
        max_total_time = max_total_time,
        output_folder_path = output_folder_path,
        output_file_name_root = output_file_name_root + '_' + similarity_type,
        metric_to_optimize = metric_to_optimize,
        cutoff_to_optimize = cutoff_to_optimize,
        recommender_input_args_last_test = recommender_input_args_last_test)

# def runHyperparameterSearch_Content(
#     recommender_class, URM_train, ICM_object, ICM_name, URM_train_last_test = None,
#     n_cases = None, n_random_starts = None, resume_from_saved = False,
#     save_model = 'best', evaluate_on_test = 'best', max_total_time = None,
#     evaluator_validation= None, evaluator_test=None, metric_to_optimize = None, cutoff_to_optimize = None,
#     output_folder_path ='result_experiments/', parallelizeKNN = False, allow_weighting = True, allow_bias_ICM = False,
#     similarity_type_list = None):
    
#     '''
#     This function performs the hyperparameter optimization for a content-based recommender

#     :param recommender_class:   Class of the recommender object to optimize, it must be a Base type
#     :param URM_train:           Sparse matrix containing the URM training data
#     :param ICM_object:          Sparse matrix containing the ICM training data
#     :param ICM_name:            String containing the name of the ICM, will be used for the name of the output files
#     :param URM_train_last_test: Sparse matrix containing the union of URM training and validation data to be used in the last evaluation
#     :param n_cases:             Number of hyperparameter sets to explore
#     :param n_random_starts:     Number of the initial random hyperparameter values to explore, usually set at 30% of n_cases
#     :param resume_from_saved:   Boolean value, if True the optimization is resumed from the saved files, if False a new one is done
#     :param save_model:          ['no', 'best', 'last'] which of the models to save, see HyperparameterTuning/SearchAbstractClass for details
#     :param evaluate_on_test:    ['all', 'best', 'last', 'no'] when to evaluate the model on the test data, see HyperparameterTuning/SearchAbstractClass for details
#     :param max_total_time:    [None or int] if set stops the hyperparameter optimization when the time in seconds for training and validation exceeds the threshold
#     :param evaluator_validation:    Evaluator object to be used for the validation of each hyperparameter set
#     :param evaluator_test:          Evaluator object to be used for the test results, the output will only be saved but not used
#     :param metric_to_optimize:  String with the name of the metric to be optimized as contained in the output of the evaluator objects
#     :param cutoff_to_optimize:  Integer with the recommendation list length to be optimized as contained in the output of the evaluator objects
#     :param output_folder_path:  Folder in which to save the output files
#     :param parallelizeKNN:      Boolean value, if True the various heuristics of the KNNs will be computed in parallel, if False sequentially
#     :param allow_weighting:     Boolean value, if True it enables the use of TF-IDF and BM25 to weight features, users and items in KNNs
#     :param allow_bias_ICM:      Boolean value, if True it enables the use of bias to shift the values of the ICM
#     :param similarity_type_list: List of strings with the similarity heuristics to be used for the KNNs
#     '''

#     # If directory does not exist, create
#     if not os.path.exists(output_folder_path):
#         os.makedirs(output_folder_path)

#     URM_train = URM_train.copy()
#     ICM_object = ICM_object.copy()

#     if URM_train_last_test is not None:
#         URM_train_last_test = URM_train_last_test.copy()

#     assert recommender_class in [ItemKNNCBF, UserKNNCBF]


#     output_file_name_root = recommender_class.RECOMMENDER_NAME + '_{}'.format(ICM_name)
#     hyperparameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

#     if similarity_type_list is None:
#         similarity_type_list = ['cosine', 'jaccard', 'asymmetric', 'dice', 'tversky']

#     recommender_input_args = SearchInputRecommenderArgs(
#         CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_object],
#         CONSTRUCTOR_KEYWORD_ARGS = {},
#         FIT_POSITIONAL_ARGS = [],
#         FIT_KEYWORD_ARGS = {}
#     )

#     if URM_train_last_test is not None:
#         recommender_input_args_last_test = recommender_input_args.copy()
#         recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
#     else:
#         recommender_input_args_last_test = None

#     run_KNNCBFRecommender_on_similarity_type_partial = partial(
#         run_KNNRecommender_on_similarity_type,
#         recommender_input_args = recommender_input_args,
#         hyperparameter_search_space = {},
#         hyperparameterSearch = hyperparameterSearch,
#         n_cases = n_cases,
#         n_random_starts = n_random_starts,
#         resume_from_saved = resume_from_saved,
#         save_model = save_model,
#         evaluate_on_test = evaluate_on_test,
#         max_total_time = max_total_time,
#         output_folder_path = output_folder_path,
#         output_file_name_root = output_file_name_root,
#         metric_to_optimize = metric_to_optimize,
#         cutoff_to_optimize = cutoff_to_optimize,
#         allow_weighting = allow_weighting,
#         allow_bias_ICM = allow_bias_ICM,
#         recommender_input_args_last_test = recommender_input_args_last_test)

#     if parallelizeKNN:
#         pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
#         pool.map(run_KNNCBFRecommender_on_similarity_type_partial, similarity_type_list)

#         pool.close()
#         pool.join()

#     else:
#         for similarity_type in similarity_type_list:
#             run_KNNCBFRecommender_on_similarity_type_partial(similarity_type)

def runHyperparameterSearch_Collaborative(
    recommender_class, URM_train, URM_train_last_test = None,
    n_cases = None, n_random_starts = None, resume_from_saved = False,
    save_model = 'best', evaluate_on_test = 'best', max_total_time = None,
    evaluator_validation = None, evaluator_test = None, evaluator_validation_earlystopping = None,
    metric_to_optimize = None, cutoff_to_optimize = None,
    output_folder_path ='result_experiments/', parallelizeKNN = True,
    allow_weighting = True, allow_bias_URM=False, allow_dropout_MF = False, similarity_type_list = None
):
    '''
    This function performs the hyperparameter optimization for a collaborative recommender

    :param recommender_class:   Class of the recommender object to optimize, it must be a Base type
    :param URM_train:           Sparse matrix containing the URM training data
    :param URM_train_last_test: Sparse matrix containing the union of URM training and validation data to be used in the last evaluation
    :param n_cases:             Number of hyperparameter sets to explore
    :param n_random_starts:     Number of the initial random hyperparameter values to explore, usually set at 30% of n_cases
    :param resume_from_saved:   Boolean value, if True the optimization is resumed from the saved files, if False a new one is done
    :param save_model:          ['no', 'best', 'last'] which of the models to save, see HyperparameterTuning/SearchAbstractClass for details
    :param evaluate_on_test:    ['all', 'best', 'last', 'no'] when to evaluate the model on the test data, see HyperparameterTuning/SearchAbstractClass for details
    :param max_total_time:    [None or int] if set stops the hyperparameter optimization when the time in seconds for training and validation exceeds the threshold
    :param evaluator_validation:    Evaluator object to be used for the validation of each hyperparameter set
    :param evaluator_validation_earlystopping:   Evaluator object to be used for the earlystopping of ML algorithms, can be the same of evaluator_validation
    :param evaluator_test:          Evaluator object to be used for the test results, the output will only be saved but not used
    :param metric_to_optimize:  String with the name of the metric to be optimized as contained in the output of the evaluator objects
    :param cutoff_to_optimize:  Integer with the recommendation list length to be optimized as contained in the output of the evaluator objects
    :param output_folder_path:  Folder in which to save the output files
    :param parallelizeKNN:      Boolean value, if True the various heuristics of the KNNs will be computed in parallel, if False sequentially
    :param allow_weighting:     Boolean value, if True it enables the use of TF-IDF and BM25 to weight features, users and items in KNNs
    :param allow_bias_URM:      Boolean value, if True it enables the use of bias to shift the values of the URM
    :param allow_dropout_MF:    Boolean value, if True it enables the use of dropout on the latent factors of MF algorithms
    :param similarity_type_list: List of strings with the similarity heuristics to be used for the KNNs
    '''

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    earlystopping_keywargs = {
        'validation_every_n': 5,
        'stop_on_validation': True,
        'evaluator_object': evaluator_validation_earlystopping,
        'lower_validations_allowed': 5,
        'validation_metric': metric_to_optimize,
    }

    URM_train = URM_train.copy()
    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()
    try:
        output_file_name_root = recommender_class.RECOMMENDER_NAME
        hyperparameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

        if recommender_class in [ItemKNNCF, UserKNNCF]:
            if similarity_type_list is None:
                similarity_type_list = ['cosine', 'jaccard', 'asymmetric', 'dice', 'tversky']

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )

            if URM_train_last_test is not None:
                recommender_input_args_last_test = recommender_input_args.copy()
                recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
            else:
                recommender_input_args_last_test = None

            run_KNNCFRecommender_on_similarity_type_partial = partial(
                run_KNNRecommender_on_similarity_type,
                recommender_input_args = recommender_input_args,
                hyperparameter_search_space = {},
                hyperparameterSearch = hyperparameterSearch,
                n_cases = n_cases,
                n_random_starts = n_random_starts,
                resume_from_saved = resume_from_saved,
                save_model = save_model,
                evaluate_on_test = evaluate_on_test,
                max_total_time = max_total_time,
                output_folder_path = output_folder_path,
                output_file_name_root = output_file_name_root,
                metric_to_optimize = metric_to_optimize,
                cutoff_to_optimize = cutoff_to_optimize,
                allow_weighting = allow_weighting,
                allow_bias_URM = allow_bias_URM,
                recommender_input_args_last_test = recommender_input_args_last_test)

            if parallelizeKNN:
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
                pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

                pool.close()
                pool.join()

            else:
                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)

            return

        if recommender_class is P3alpha:
            hyperparameters_range_dictionary = {
                'topK': Integer(5, 1000),
                'alpha': Real(low = 0, high = 2, prior = 'uniform'),
                'normalize_similarity': Categorical([True, False]),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )

        if recommender_class is RP3beta:
            hyperparameters_range_dictionary = {
                'topK': Integer(5, 1000),
                'alpha': Real(low = 0, high = 2, prior = 'uniform'),
                'beta': Real(low = 0, high = 2, prior = 'uniform'),
                'normalize_similarity': Categorical([True, False]),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )

        if recommender_class is PureSVD:
            hyperparameters_range_dictionary = {
                'num_factors': Integer(1, 350),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )

        if recommender_class is PureSVDItem:
            hyperparameters_range_dictionary = {
                'num_factors': Integer(1, 350),
                'topK': Integer(5, 1000),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )

        if recommender_class is SLIM_BPR:
            hyperparameters_range_dictionary = {
                'topK': Integer(5, 1000),
                'epochs': Categorical([1500]),
                'symmetric': Categorical([True, False]),
                'sgd_mode': Categorical(['sgd', 'adagrad', 'adam']),
                'lambda_i': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
                'lambda_j': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
                'learning_rate': Real(low = 1e-4, high = 1e-1, prior = 'log-uniform'),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {
                    **earlystopping_keywargs,
                }
            )

        if recommender_class is SLIMElasticNet or recommender_class is MultiThreadSLIM_SLIMElasticNet:
            hyperparameters_range_dictionary = {
                'alpha': Real(low = 1e-3, high = 1.0, prior = 'uniform'),
                'l1_ratio': Real(low = 1e-5, high = 1.0, prior = 'log-uniform'),
                'topK': Integer(5, 1000),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )

        if recommender_class is EASE_R:
            hyperparameters_range_dictionary = {
                'topK': Categorical([None]), #Integer(5, 3000),
                'l2_norm': Real(low = 1e0, high = 1e7, prior = 'log-uniform'),
                'normalize_matrix': Categorical([False]), # With normalize_matrix:True tends to perform worse
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )

        # if recommender_class is MatrixFactorization_FunkSVD_Cython:
        #     hyperparameters_range_dictionary = {
        #         'sgd_mode': Categorical(['sgd', 'adagrad', 'adam']),
        #         'epochs': Categorical([500]),
        #         'use_bias': Categorical([True, False]),
        #         'batch_size': Categorical([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
        #         'num_factors': Integer(1, 200),
        #         'item_reg': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
        #         'user_reg': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
        #         'learning_rate': Real(low = 1e-4, high = 1e-1, prior = 'log-uniform'),
        #         'negative_interactions_quota': Real(low = 0.0, high = 0.5, prior = 'uniform'),
        #     }

        #     if allow_dropout_MF:
        #         hyperparameters_range_dictionary['dropout_quota'] = Real(low = 0.01, high = 0.7, prior = 'uniform')

        #     recommender_input_args = SearchInputRecommenderArgs(
        #         CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
        #         CONSTRUCTOR_KEYWORD_ARGS = {},
        #         FIT_POSITIONAL_ARGS = [],
        #         FIT_KEYWORD_ARGS = earlystopping_keywargs
        #     )
            
        # if recommender_class is MatrixFactorization_AsySVD_Cython:
        #     hyperparameters_range_dictionary = {
        #         'sgd_mode': Categorical(['sgd', 'adagrad', 'adam']),
        #         'epochs': Categorical([500]),
        #         'use_bias': Categorical([True, False]),
        #         'batch_size': Categorical([1]),
        #         'num_factors': Integer(1, 200),
        #         'item_reg': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
        #         'user_reg': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
        #         'learning_rate': Real(low = 1e-4, high = 1e-1, prior = 'log-uniform'),
        #         'negative_interactions_quota': Real(low = 0.0, high = 0.5, prior = 'uniform'),
        #     }

        #     recommender_input_args = SearchInputRecommenderArgs(
        #         CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
        #         CONSTRUCTOR_KEYWORD_ARGS = {},
        #         FIT_POSITIONAL_ARGS = [],
        #         FIT_KEYWORD_ARGS = earlystopping_keywargs
        #     )

        # if recommender_class is MatrixFactorization_BPR_Cython:
        #     hyperparameters_range_dictionary = {
        #         'sgd_mode': Categorical(['sgd', 'adagrad', 'adam']),
        #         'epochs': Categorical([1500]),
        #         'num_factors': Integer(1, 200),
        #         'batch_size': Categorical([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
        #         'positive_reg': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
        #         'negative_reg': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
        #         'learning_rate': Real(low = 1e-4, high = 1e-1, prior = 'log-uniform'),
        #     }

        #     if allow_dropout_MF:
        #         hyperparameters_range_dictionary['dropout_quota'] = Real(low = 0.01, high = 0.7, prior = 'uniform')

        #     recommender_input_args = SearchInputRecommenderArgs(
        #         CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
        #         CONSTRUCTOR_KEYWORD_ARGS = {},
        #         FIT_POSITIONAL_ARGS = [],
        #         FIT_KEYWORD_ARGS = {**earlystopping_keywargs}
        #     )

        if recommender_class is IALS:
            hyperparameters_range_dictionary = {
                'num_factors': Integer(1, 200),
                'epochs': Categorical([300]),
                'confidence_scaling': Categorical(['linear', 'log']),
                'alpha': Real(low = 1e-3, high = 50.0, prior = 'log-uniform'),
                'epsilon': Real(low = 1e-3, high = 10.0, prior = 'log-uniform'),
                'reg': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = earlystopping_keywargs
            )

        # if recommender_class is NMFRecommender:
        #     hyperparameters_range_dictionary = {
        #         'num_factors': Integer(1, 350),
        #         'solver': Categorical(['coordinate_descent', 'multiplicative_update']),
        #         'init_type': Categorical(['random', 'nndsvda']),
        #         'beta_loss': Categorical(['frobenius', 'kullback-leibler']),
        #     }

        #     recommender_input_args = SearchInputRecommenderArgs(
        #         CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
        #         CONSTRUCTOR_KEYWORD_ARGS = {},
        #         FIT_POSITIONAL_ARGS = [],
        #         FIT_KEYWORD_ARGS = {}
        #     )

        if recommender_class is LightFMCF:
            hyperparameters_range_dictionary = {
                'epochs': Categorical([300]),
                'n_components': Integer(1, 200),
                'loss': Categorical(['bpr', 'warp', 'warp-kos']),
                'sgd_mode': Categorical(['adagrad', 'adadelta']),
                'learning_rate': Real(low = 1e-6, high = 1e-1, prior = 'log-uniform'),
                'item_alpha': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
                'user_alpha': Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = earlystopping_keywargs
            )

        # if recommender_class is MultVAERecommender:
        #     n_items = URM_train.shape[1]

        #     hyperparameters_range_dictionary = {
        #         'epochs': Categorical([300]),
        #         'learning_rate': Real(low=1e-6, high=1e-2, prior='log-uniform'),
        #         'l2_reg': Real(low=1e-6, high=1e-2, prior='log-uniform'),
        #         'dropout': Real(low=0., high=0.8, prior='uniform'),
        #         'total_anneal_steps': Integer(100000, 600000),
        #         'anneal_cap': Real(low=0., high=0.6, prior='uniform'),
        #         'batch_size': Categorical([128, 256, 512, 1024]),

        #         'encoding_size': Integer(1, min(512, n_items)),
        #         'next_layer_size_multiplier': Integer(2, 10),
        #         'max_n_hidden_layers': Integer(1, 4),
        #         # Reduce max_layer_size if estimated last layer weights size exceeds 2 GB
        #         'max_layer_size': Categorical([min(5*1e3, int(2*1e9*8/64/n_items))]),
        #     }

        if URM_train_last_test is not None:
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        else:
            recommender_input_args_last_test = None

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        hyperparameterSearch.search(
            recommender_input_args,
            hyperparameter_search_space= hyperparameters_range_dictionary,
            n_cases = n_cases,
            n_random_starts = n_random_starts,
            resume_from_saved = resume_from_saved,
            save_model = save_model,
            evaluate_on_test = evaluate_on_test,
            max_total_time = max_total_time,
            output_folder_path = output_folder_path,
            output_file_name_root = output_file_name_root,
            metric_to_optimize = metric_to_optimize,
            cutoff_to_optimize = cutoff_to_optimize,
            recommender_input_args_last_test = recommender_input_args_last_test)

    except Exception as e:
        print('On recommender {} Exception {}'.format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + 'ErrorLog.txt', 'a')
        error_file.write('On recommender {} Exception {}\n'.format(recommender_class, str(e)))
        error_file.close()