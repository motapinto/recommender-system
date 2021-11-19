from Utils.Dataset import Dataset
from Utils.Evaluator import Evaluator
from Utils.Evaluator import EvaluatorHoldout
from Utils.methods.get_recommender_class import get_recommender_class

if __name__ == '__main__':
    dataset = Dataset(path='./Data', validation=0.1, testing=0.1)

    evaluator_validation = EvaluatorHoldout(dataset.URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(dataset.URM_test, cutoff_list=[10])

    models = ['ItemKNNCFRecommender', 'UserKNNCFRecommender']
    model_name = models[1]

    recommender_class = get_recommender_class(model_name)
    recommender = recommender_class(dataset.URM_train)
    recommender.fit()

    _, results_run_string = evaluator_validation.evaluateRecommender(recommender)
    print(results_run_string)
    
    _, results_run_string = evaluator_test.evaluateRecommender(recommender)
    print(results_run_string)