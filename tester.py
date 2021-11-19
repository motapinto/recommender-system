import time

from Utils.Dataset import Dataset
from Utils.Evaluator import EvaluatorHoldout
from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation=0.1, testing=0.1)

    recommender = SLIM_BPR_Python(dataset.URM_train)
    recommender.fit()

    evaluator_validation = EvaluatorHoldout(dataset.URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(dataset.URM_test, cutoff_list=[10])

    results_df, results_run_string = evaluator_validation.evaluateRecommender(recommender)
    val_map, val_precision, val_recall = results_df.loc[10, 'MAP'], results_df.loc[10, 'PRECISION'], results_df.loc[10, 'RECALL']
    #print('\nValidation results')
    # print(results_run_string)
    
    results_df, results_run_string = evaluator_test.evaluateRecommender(recommender)
    test_map, test_precision, test_recall = results_df.loc[10, 'MAP'], results_df.loc[10, 'PRECISION'], results_df.loc[10, 'RECALL']
    #print('Test results')
    # print(results_run_string)

    print(f'\nVal MAP: {val_map} | Val PRECISION: {val_precision} | Val RECALL: {val_recall}')
    print(f'Test MAP: {test_map} | Test PRECISION: {test_precision} | Test RECALL: {val_recall}')
