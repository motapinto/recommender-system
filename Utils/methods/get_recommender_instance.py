from Recommenders.CB.KNN.ItemKNNCBF import ItemKNNCBF

def get_recommender_inputs(recommender_class, URM_train, ICM_train):
    if recommender_class is ItemKNNCBF:
        return URM_train, ICM_train
    
    return URM_train