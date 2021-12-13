import os
import traceback
from functools import partial
from skopt.space import Integer, Categorical
from skopt.space.space import Real

from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Utils.methods.get_recommender_instance import get_recommender_instance

# Hyper-parameters
from Recommenders.Search.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.Search.run_hyperparameter_search import runHyperparameterSearch_Hybrid
from Recommenders.Search.run_hyperparameter_search import runHyperparameterSearch_Content
from Recommenders.Search.SearchAbstractClass import SearchInputRecommenderArgs
from Recommenders.Search.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.Similarity.Compute_Similarity import SimilarityFunction

# Recommenders
from Utils.import_recommenders import *
from Recommenders.CF.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython,\
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython

def run_search(hyperparameter_search_cf, cf_models):
    for recommender in cf_models:
        try: hyperparameter_search_cf(recommender)
        except Exception as e:
            print('On recommender {} Exception {}'.format(recommender, str(e)))
            traceback.print_exc()

def tune_cf(
    URM_train, URM_train_val, evaluator_validation, evaluator_test, 
    cf_models, output_folder_path, n_cases=20
): 
    run_hyperparameter_search_cf = partial(runHyperparameterSearch_Collaborative,
        URM_train=URM_train,
        URM_train_last_test=URM_train_val,
        metric_to_optimize='MAP',
        cutoff_to_optimize=10,
        evaluator_validation_earlystopping=evaluator_validation,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test,
        output_folder_path=output_folder_path,
        parallelizeKNN=True,
        allow_weighting=True,
        allow_dropout_MF=True,
        allow_bias_URM=True,
        resume_from_saved=False,
        similarity_type_list= [
            #SimilarityFunction.ASYMMETRIC, 
            #SimilarityFunction.TVERSKY,
            'adjusted',
        ],
        save_model='no',
        n_cases=n_cases,
        n_random_starts=int(n_cases*0.3))

    run_search(run_hyperparameter_search_cf, cf_models)

def tune_hybrid(
    URM_train, URM_train_val, ICM_object,
    evaluator_validation, evaluator_test, 
    cf_models, output_folder_path, n_cases=20
): 
    run_hyperparameter_search_cf = partial(runHyperparameterSearch_Hybrid,
        URM_train=URM_train,
        URM_train_last_test=URM_train_val,
        ICM_object=ICM_object, 
        ICM_name='',
        metric_to_optimize='MAP',
        cutoff_to_optimize=10,
        evaluator_validation_earlystopping=evaluator_validation,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test,
        output_folder_path=output_folder_path,
        parallelizeKNN=True,
        allow_weighting=True,
        resume_from_saved=False,
        similarity_type_list=['asymmetric', 'tversky'],
        save_model='no',
        n_cases=n_cases,
        n_random_starts=int(n_cases*0.3))

    run_search(run_hyperparameter_search_cf, cf_models)

def tune_cbf(
    URM_train, URM_train_val, ICM_object,
    evaluator_validation, evaluator_test, 
    cf_models, output_folder_path, n_cases=20
): 
    run_hyperparameter_search_cf = partial(runHyperparameterSearch_Content,
        URM_train=URM_train,
        URM_train_last_test=URM_train_val,
        ICM_object=ICM_object, 
        ICM_name='',
        metric_to_optimize='MAP',
        cutoff_to_optimize=10,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test,
        output_folder_path=output_folder_path,
        parallelizeKNN=True,
        allow_weighting=True,
        allow_bias_ICM=True,
        resume_from_saved=False,
        similarity_type_list=['asymmetric', 'tversky'], # only for ItemKNNCF, UserKNNCF
        save_model='no',
        n_cases=n_cases,
        n_random_starts=int(n_cases*0.3))

    run_search(run_hyperparameter_search_cf, cf_models)

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation_percentage=0.1, test_percentage=0.2)

    ICM = dataset.get_icm_format_k(11)
    stacked_URM, _ = dataset.stack_URM_ICM(dataset.URM_train, ICM)

    evaluator_validation = EvaluatorHoldout(dataset.URM_val, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(dataset.URM_test, cutoff_list=[10])

    output_folder_path = os.path.join('Recommenders', 'tuner_results'+os.sep)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    cf_models = [
        # ItemKNNCF,
        # UserKNNCF,
        # RP3beta,
        # P3alpha,
        # EASE_R,
        
        # SLIM_BPR, ---- today -- check samples per second
        # SLIMElasticNet,
        MultiThreadSLIM_SLIMElasticNet
        
        # PureSVD,
        # ScaledPureSVD,
        # PureSVDItem,
        #IALS,  ---- today -- check samples per second
        # LightFM, ---- today -- check samples per second
        #MatrixFactorization_FunkSVD_Cython,
        #MatrixFactorization_AsySVD_Cython, --- check
        #MatrixFactorization_BPR_Cython --- check
    ]


    tune_cf(stacked_URM, dataset.URM_train_val, evaluator_validation, 
        evaluator_test, cf_models, output_folder_path, n_cases=50)

    # tune_cbf(dataset.URM_train, dataset.URM_train_val, dataset.ICM,
    #     evaluator_validation, evaluator_test, [ItemKNNCBF], output_folder_path, n_cases=200)

    # hybrid_models = [Hybrid2]
    # tune_hybrid(dataset.URM_train, dataset.URM_train_val, dataset.ICM, 
    #     evaluator_validation, evaluator_test, hybrid_models, output_folder_path, n_cases=20)
    
    # hyperparameters = {
    #     'topK': Integer(low=1e2, high=2e3, prior='uniform', base=10),
    #     'l2_norm': Real(low=1e3, high=1e5, prior='log-uniform'),
    #     'normalize_matrix': Categorical([False]), # With normalize_matrix:True tends to perform worse
    # }
    # tune_one(EASE_R, hyperparameters, evaluator_validation, evaluator_test, output_folder_path, n_cases=60)

# earlystopping_keywargs = {}
#         if recommender in [IALS, SLIM_BPR]:
#             earlystopping_keywargs = {
#                 'validation_every_n': 5,
#                 'stop_on_validation': True,
#                 'evaluator_object': EvaluatorHoldout(dataset.URM_test, cutoff_list=[10]),
#                 'lower_validations_allowed': 5,
#                 'validation_metric': 'MAP',
#             } 
        
#         recommender.fit(earlystopping_keywargs)