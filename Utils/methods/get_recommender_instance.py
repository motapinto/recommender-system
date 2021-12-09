from Recommenders.CB.KNN.ItemKNNCBF import ItemKNNCBF
from Recommenders.Hybrid.ItemKNN_CFCBF_Hybrid import ItemKNN_CFCBF_Hybrid

def get_recommender_instance(recommender_class, URM_train, ICM_train):
    if recommender_class is [ItemKNNCBF, ItemKNN_CFCBF_Hybrid]:
        return recommender_class(URM_train, ICM_train)
    
    return recommender_class(URM_train)