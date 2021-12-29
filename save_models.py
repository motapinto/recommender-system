import os
from Utils.Dataset import Dataset
from Utils.import_recommenders import *
from Utils.methods.get_recommender_instance import get_recommender_instance

def save_models(URM, ICM, output_folder_path):
    models = [
        EASE_R, 
        MultiThreadSLIM_SLIMElasticNet,
        ItemKNNCF,
        UserKNNCF,
        PureSVDItem,
        PureSVD,
        RP3beta,
        ItemKNN_CFCBF_Hybrid,
    ]

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for model in models:
        recommender = get_recommender_instance(model, URM, ICM)
        recommender.fit()
        recommender.save_model(output_folder_path, recommender.RECOMMENDER_NAME)

if __name__ == '__main__':
    prod_dataset = Dataset(path='./Data', validation_percentage=0, test_percentage=0, seed=None)
    stacked_URM, _ = prod_dataset.stack_URM_ICM(prod_dataset.URM_train, prod_dataset.ICM)
    output_folder_path = os.path.join('Recommenders', 'saved_models', 'prod'+os.sep)
    save_models(stacked_URM, prod_dataset.ICM, output_folder_path)

    test_dataset = Dataset(path='./Data', validation_percentage=0, test_percentage=0.2, seed=1234)
    stacked_URM, _ = test_dataset.stack_URM_ICM(test_dataset.URM_train, test_dataset.ICM)
    output_folder_path = os.path.join('Recommenders', 'saved_models', 'test'+os.sep)
    save_models(stacked_URM, test_dataset.ICM, output_folder_path)
