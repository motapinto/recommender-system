import os
import traceback
from functools import partial

from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Utils.methods.get_recommender_instance import get_recommender_instance

from Recommenders.Search.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.Search.run_hyperparameter_search import runHyperparameterSearch_Hybrid
from Recommenders.Search.run_hyperparameter_search import runHyperparameterSearch_Content

from Utils.import_recommenders import *

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
            'asymmetric', 
            'tversky',
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
    dataset = Dataset(path='./Data', validation_percentage=0.1, test_percentage=0.2, seed=1234)
    stacked_URM, stacked_ICM = dataset.stack_URM_ICM(dataset.URM_train, dataset.ICM)

    evaluator_validation = EvaluatorHoldout(dataset.URM_val, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(dataset.URM_test, cutoff_list=[10])

    output_folder_path = os.path.join('Recommenders', 'tuner_results'+os.sep)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # cf_models = []
    # tune_cf(stacked_URM, dataset.URM_train_val, evaluator_validation, 
    #     evaluator_test, cf_models, output_folder_path, n_cases=100)

    cf_models = [ItemKNNCF]
    tune_cf(stacked_URM, dataset.URM_train_val, evaluator_validation, 
        evaluator_test, cf_models, output_folder_path, n_cases=400)

    # cbf_models = []
    # tune_cbf(dataset.URM_train, dataset.URM_train_val, dataset.ICM,
    #     evaluator_validation, evaluator_test, cbf_models, output_folder_path, n_cases=100)

    # hybrid_models = [Hybrid]
    # tune_hybrid(stacked_URM, dataset.URM_train_val, dataset.ICM, 
    #     evaluator_validation, evaluator_test, hybrid_models, output_folder_path, n_cases=100)
    