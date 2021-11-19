import os
from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Recommenders.SLIM.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.search.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.search.SearchAbstractClass import SearchInputRecommenderArgs

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation=0.1, testing=0.1)

    evaluator_validation = EvaluatorHoldout(dataset.URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(dataset.URM_test, cutoff_list=[10])

    parameterSearch = SearchBayesianSkopt(
        SLIM_BPR_Cython,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test)

    output_folder_path = "result_experiments/"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [dataset.URM_train, dataset.ICM],
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = {}
    )

    parameterSearch.search(
        recommender_input_args,
        parameter_search_space = {}, # hyperparameters: arguments from fit function
        n_cases = 200,
        n_random_starts = 20,
        save_model="no",
        output_folder_path = output_folder_path,
        output_file_name_root = recommender_class.RECOMMENDER_NAME,
        metric_to_optimize = "MAP")