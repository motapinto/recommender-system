import os
import traceback
from functools import partial
from skopt.space import Integer, Categorical

from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Utils.methods.get_recommender_instance import get_recommender_inputs

# Hyper-parameters
from Recommenders.Search.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.Search.SearchAbstractClass import SearchInputRecommenderArgs
from Recommenders.Search.SearchBayesianSkopt import SearchBayesianSkopt

# CF
from Recommenders.CF.KNN.ItemKNNCF import ItemKNNCF
from Recommenders.CF.KNN.UserKNNCF import UserKNNCF
from Recommenders.CF.KNN.RP3beta import RP3beta
from Recommenders.CF.KNN.P3alpha import P3alpha
from Recommenders.CF.KNN.EASE_R import EASE_R
from Recommenders.CF.KNN.MachineLearning.SLIM_BPR import SLIM_BPR
from Recommenders.CF.KNN.MachineLearning.SLIMElasticNet import SLIMElasticNet
from Recommenders.CF.MatrixFactorization.PureSVD import PureSVD, ScaledPureSVD
from Recommenders.CF.MatrixFactorization.PureSVDItem import PureSVDItem
from Recommenders.CF.MatrixFactorization.IALS import IALS
from Recommenders.CF.MatrixFactorization.LightFM import LightFMCF

def run_search(hyperparameter_search_cf, cf_models):
    for recommender in cf_models:
        try: hyperparameter_search_cf(recommender)
        except Exception as e:
            print('On recommender {} Exception {}'.format(recommender, str(e)))
            traceback.print_exc()

def tune_all_cf(
    URM_train, URM_train_val, evaluator_validation, evaluator_test, cf_models,
    n_cases=20, output_folder_path=os.path.join('Recommenders', 'tuner_results'+os.sep)
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
       resume_from_saved=True,
       save_model='best',
       similarity_type_list=['cosine', 'jaccard', 'asymmetric', 'dice', 'tversky'], # only for ItemKNNCF, UserKNNCF
       n_cases=n_cases,
       n_random_starts=int(n_cases*0.3))

    run_search(run_hyperparameter_search_cf, cf_models)

def tune_all_cb():
    run_hyperparameter_search_cf = None
    run_search(run_hyperparameter_search_cf, cf_models)

def tune_one(
    recommender_class, hyperparameters, evaluator_validation, evaluator_test, 
    n_cases=20, output_folder_path=os.path.join('Recommenders', 'tuner_results'+os.sep)
):
    args = list(get_recommender_inputs(recommender_class, dataset.URM_train, dataset.ICM))

    parameterSearch = SearchBayesianSkopt(
        recommender_class,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=args,
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    parameterSearch.search(
        recommender_input_args,
        hyperparameter_search_space=hyperparameters,
        n_cases=n_cases,
        n_random_starts=Integer(n_cases*0.3),
        save_model='no',
        output_folder_path=output_folder_path,
        output_file_name_root=recommender_class.RECOMMENDER_NAME,
        metric_to_optimize='MAP',
        cutoff_to_optimize=10 )

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation_percentage=0.1, test_percentage=0.1)

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
        # SLIM_BPR,
        # SLIMElasticNet,
        # PureSVD,
        # ScaledPureSVD, -> cannot be used in tune_all() -> lacks implementation (can be used in tune_one())
        # PureSVDItem,
        # IALS,
        LightFMCF,
    ]

    tune_all_cf(dataset.URM_train, dataset.URM_train_val, evaluator_validation, 
        evaluator_test, cf_models, n_cases=20)
    
    # hyperparameters = {
    #     'topK': Integer(5, 1000),
    #     'shrink': Integer(0, 1000),
    #     'similarity': Categorical(['cosine']),
    #     'normalize': Categorical([True, False]),
    # }
    # tune_one(SLIM_BPR, hyperparameters, n_cases=100)
    
# SLIM -> SLIMElasticNet -> ScaledPureSVD?