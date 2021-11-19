import os
import numpy as np
from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Utils.methods.get_recommender_class import get_recommender_class
from Recommenders.CF.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.CF.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.Search.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.Search.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Integer, Categorical


def define_input_args(model_name, dataset, all=False):
    if model_name in ['ItemKNNCFRecommender', 'UserKNNCFRecommender', 'SLIM_BPR_Recommender']: # CF
        if not all: return [dataset.URM_train]
        else: return [] # train + val
    else: # CB
        return [dataset.ICM['...']]

def get_hyperparameters_range_dict(model_name):
    if model_name in ['ItemKNNCFRecommender', 'UserKNNCFRecommender']:
        return {
            'topK': Integer(5, 1000),
            'shrink': Integer(0, 1000),
            'similarity': Categorical(['cosine', 'pearson', 'jaccard', 'tanimoto', 'adjusted', 'euclidean']),
            'normalize': Categorical([True, False]),
            'feature_weighting': Categorical(['BM25', 'TF-IDF', 'none']),
            'URM_bias': Categorical([True, False]),
        }

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation=0.1, testing=0.1)

    evaluator_validation = EvaluatorHoldout(dataset.URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(dataset.URM_test, cutoff_list=[10])

    models = ['ItemKNNCFRecommender', 'UserKNNCFRecommender']
    model_name = models[1]

    parameterSearch = SearchBayesianSkopt(
        get_recommender_class(model_name),
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = define_input_args(model_name, dataset),
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = {})

    output_folder_path = os.path.join('Recommenders', 'tuner_results'+os.sep)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    parameterSearch.search(
        recommender_input_args,
        hyperparameter_search_space=get_hyperparameters_range_dict(model_name), 
        n_cases = 20,
        n_random_starts = int(20*0.3),
        save_model='no',
        output_folder_path = output_folder_path,
        output_file_name_root = model_name,
        metric_to_optimize = 'MAP',
        cutoff_to_optimize=10)