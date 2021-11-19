from Recommenders.CF.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.CF.KNN.UserKNNCFRecommender import UserKNNCFRecommender

def get_recommender_class(model_name):
    if model_name == 'ItemKNNCFRecommender': return ItemKNNCFRecommender
    if model_name == 'UserKNNCFRecommender': return UserKNNCFRecommender
