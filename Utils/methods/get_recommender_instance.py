import os
from Utils.import_recommenders import *

def get_recommender_instance(recommender_class, URM_train, ICM_train):
    if recommender_class in [ItemKNNCBF, ItemKNN_CFCBF_Hybrid, Hybrid1, Hybrid2, Hybrid3]:
        return recommender_class(URM_train, ICM_train)

    elif recommender_class in [Hybrid4]:
        return recommender_class(URM_train, ICM_train, 'test') 

    return recommender_class(URM_train)