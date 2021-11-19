import os
from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python
from Recommenders.Search.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.Search.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation=0.1, testing=0.1)

    evaluator_validation = EvaluatorHoldout(dataset.URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(dataset.URM_test, cutoff_list=[10])

    parameterSearch = SearchBayesianSkopt(
        SLIM_BPR_Python,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test)

    output_folder_path = 'result_experiments/'
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [dataset.URM_train],
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = {}
    )

    parameterSearch.search(
        recommender_input_args,
        hyperparameter_search_space = {
            'lambda_i': Real(1e-4, 1e-2), 
            'lambda_j': Real(1e-5, 1e-3),
            'learning_rate': Real(1e-3, 1e-1),
            }, 
        n_cases = 20,
        n_random_starts = int(20*0.3),
        save_model='no',
        output_folder_path = output_folder_path,
        output_file_name_root = SLIM_BPR_Python.RECOMMENDER_NAME,
        metric_to_optimize = 'MAP',
        cutoff_to_optimize=10 )