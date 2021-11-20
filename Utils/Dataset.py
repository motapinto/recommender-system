import os
import pandas as pd
import numpy as np
from scipy import sparse as sps
from sklearn.model_selection import train_test_split
from Utils.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

class Dataset(object):
    def __init__(self, path='./Data', train_percentage=0.8, val_split=False):
        URM = self.read_URM(path)

        num_users = len(URM['row'].unique())
        num_items = len(URM['col'].unique())

        targets = self.read_targets(path)
        self.targets = np.unique(targets['user_id'])

        ICM = self.read_ICM(path)
        self.ICM = { 
            'channel_ICM': sps.csr_matrix((ICM['channel_ICM']['data'], (ICM['channel_ICM']['row'], ICM['channel_ICM']['col']))), 
            'event_ICM': sps.csr_matrix((ICM['event_ICM']['data'], (ICM['event_ICM']['row'], ICM['event_ICM']['col']))), 
            'genre_ICM': sps.csr_matrix((ICM['genre_ICM']['data'], (ICM['genre_ICM']['row'], ICM['genre_ICM']['col']))), 
            'subgenre_ICM': sps.csr_matrix((ICM['subgenre_ICM']['data'], (ICM['subgenre_ICM']['row'], ICM['subgenre_ICM']['col'])))
        }

        URM = sps.csr_matrix((URM['data'], (URM['row'], URM['col'])), shape = (num_users, num_items))
        
        self.URM_train_val, self.URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=train_percentage)
        self.URM_train = self.URM_train_val
        if val_split: self.URM_train, self.URM_val = split_train_in_two_percentage_global_sample(self.URM_train_val, train_percentage=train_percentage)

    def split(self, data, validation=0.1, test=0.1):
        seed = 9999
        user_train = data['row'] 
        item_train = data['col'] 
        rating_train = data['data'] 

        split_train_in_two_percentage_global_sample(data, train_percentage = 0.8)

        (user_train, user_val,
        item_train, item_val,
        rating_train, rating_val) = train_test_split(
            data['row'],
            data['col'],
            data['data'],
            test_size=validation+test,
            random_state=seed
        )

        (user_val, user_ids_test,
        item_val, item_ids_test,
        rating_val, ratings_test) = train_test_split(
            user_val,
            item_val,
            rating_val,
            test_size=0.5,
            shuffle=True,
            random_state=seed
        )

        self.URM_train = sps.csr_matrix(
            (rating_train, (user_train, item_train)), 
            shape=(self.num_users, self.num_items)
        )

        self.URM_validation = sps.csr_matrix(
            (rating_val, (user_val, item_val)), 
            shape=(self.num_users, self.num_items)
        )
        
        self.URM_test = sps.csr_matrix(
            (ratings_test, (user_ids_test, item_ids_test)), 
            shape=(self.num_users, self.num_items)
        )

    def read_ICM(self, path):
        channel_ICM = pd.read_csv(
            os.path.join(path, 'data_ICM_channel.csv'),
            sep=',',
            names=['row', 'col', 'data'],
            header=0,
            dtype={'row': np.int32, 'col': np.int32, 'data': np.float})
        
        event_ICM = pd.read_csv(
            os.path.join(path, 'data_ICM_event.csv'),
            sep=',',
            names=['row', 'col', 'data'],
            header=0,
            dtype={'row': np.int32, 'col': np.int32, 'data': np.float})

        genre_ICM = pd.read_csv(
            os.path.join(path, 'data_ICM_genre.csv'),
            sep=',',
            names=['row', 'col', 'data'],
            header=0,
            dtype={'row': np.int32, 'col': np.int32, 'data': np.float})

        subgenre_ICM = pd.read_csv(
            os.path.join(path, 'data_ICM_genre.csv'),
            sep=',',
            names=['row', 'col', 'data'],
            header=0,
            dtype={'row': np.int32, 'col': np.int32, 'data': np.float})

        ICM = { 
            'channel_ICM': channel_ICM, 
            'event_ICM': event_ICM, 
            'genre_ICM': genre_ICM, 
            'subgenre_ICM': subgenre_ICM
        }

        return ICM

    def read_URM(self, path):
        URM = pd.read_csv(os.path.join(path, 'data_train.csv'),
            sep=',',
            names=['row', 'col', 'data'],
            header=0,
            dtype={'row': np.int32, 'col': np.int32, 'data': np.int32}
        )

        return URM

    def read_targets(self, path):
        targets = pd.read_csv(os.path.join(path, 'data_target_users_test.csv'),
            sep=',',
            names=['user_id'],
            header=0,
            dtype={'user_id': np.int32}
        )

        return targets

    