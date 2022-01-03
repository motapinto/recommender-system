from tqdm import trange
from numpy.lib.function_base import average

from Utils.Dataset import Dataset
from Utils.import_recommenders import *
from Utils.methods.get_recommender_instance import get_recommender_instance

def evaluate_models(k, URM_trains, ICM, URM_tests, test_models):
    avg_results = []
    for model in test_models:
        result_array = []

        for i in trange(k):     
            stacked_URM, _ = dataset.stack_URM_ICM(URM_trains[i], ICM)           
            result_df, exec_time = evaluate_recommender(model, 
                stacked_URM, ICM, URM_tests[i])
            
            result_array.append(result_df.loc[10]['MAP'])
            print('\nRecommender performance: MAP = {:.4f}. Time: {} s.\n'.format(
                result_df.loc[10]['MAP'], exec_time))
        
        avg_results.append(average(result_array))

    for model_idx, map in enumerate(avg_results):
        print(f'Recommender: {test_models[model_idx].RECOMMENDER_NAME} | MAP@10: {map}')

if __name__ == '__main__':
    dataset = Dataset(path='./Data', k=0, validation_percentage=0, test_percentage=0.2, seed=1234)

    if dataset.cross_val:
        test_models = [
            #ItemKNN_CFCBF_Hybrid,
            #UserKNNCF,
            #sItemKNNCF,
            #EASE_R,
            #RP3beta
        ]

        evaluate_models(dataset.k, dataset.URM_trains, dataset.ICM, dataset.URM_tests, test_models)

    else:
        stacked_URM, stacked_ICM = dataset.stack_URM_ICM(dataset.URM_train.copy(), dataset.ICM.copy())  
        
        fit_params = {}
        model = get_recommender_instance(Hybrid, stacked_URM, dataset.ICM)
        model.evaluate_model(dataset.URM_test, fit_params, load=False)



