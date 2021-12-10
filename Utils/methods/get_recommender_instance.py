from Recommenders.CB.KNN.ItemKNNCBF import ItemKNNCBF
from Recommenders.Hybrid.ItemKNN_CFCBF_Hybrid import ItemKNN_CFCBF_Hybrid
from Recommenders.Hybrid.Hybrid1 import Hybrid1
from Recommenders.Hybrid.Hybrid2 import Hybrid2

def get_recommender_instance(recommender_class, URM_train, ICM_train):
    print(recommender_class)
    if recommender_class in [ItemKNNCBF, ItemKNN_CFCBF_Hybrid, Hybrid1, Hybrid2]:
        return recommender_class(URM_train, ICM_train)

    return recommender_class(URM_train)