import numpy as np
from tqdm import tqdm
from Recommenders.Base.Base import Base

def get_submission(targets: np.array, recommender: Base):
    recommendations_array = recommender.recommend(user_id_array=targets, cutoff=10)
    return recommendations_array

def save_submission(recommendations_array):
    with open('./submission.csv', 'w+') as f:
        f.write('user_id,item_list\n')
        for user_id, items in enumerate(recommendations_array):
            items_str = f' '.join([str(item) for item in items])
            f.write(f'{user_id},{items_str}\n')