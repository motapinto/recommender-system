import os
import similaripy as sim
import pandas as pd
import numpy as np
from scipy import sparse as sps
from sklearn.model_selection import train_test_split
from Utils.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

class Dataset(object):
    def __init__(self, path='./Data', validation_percentage=0.1, test_percentage=0.2):
        # Note: no need to create new mapping, since all item identifiers are present in the URM
        URM = self.read_URM(path)

        unique_users = URM['row'].unique()
        self.num_users = len(unique_users)
        unique_items = URM['col'].unique()
        self.num_items = len(unique_items)
        
        self.split(URM, validation_percentage, test_percentage)

        targets = self.read_targets(path)
        self.targets = np.unique(targets['user_id'])

        ICM = self.read_ICM(path)
        self.ICM = { 
            'channel_ICM': sps.csr_matrix((ICM['channel_ICM']['data'], (ICM['channel_ICM']['row'], ICM['channel_ICM']['col']))), 
            'event_ICM': sps.csr_matrix((ICM['event_ICM']['data'], (ICM['event_ICM']['row'], ICM['event_ICM']['col']))), 
            'genre_ICM': sps.csr_matrix((ICM['genre_ICM']['data'], (ICM['genre_ICM']['row'], ICM['genre_ICM']['col']))), 
            'subgenre_ICM': sps.csr_matrix((ICM['subgenre_ICM']['data'], (ICM['subgenre_ICM']['row'], ICM['subgenre_ICM']['col'])))
        }
    
    def split(self, data, validation_percentage=0.1, test_percentage=0.1):
        seed = 9999
        user_train = data['row'] 
        item_train = data['col'] 
        rating_train = data['data'] 

        (user_train, user_test,
        item_train, item_test,
        rating_train, rating_test) = train_test_split(
            data['row'],
            data['col'],
            data['data'],
            test_size=test_percentage,
            random_state=seed
        )

        if validation_percentage > 0:
            self.URM_train_val = sps.csr_matrix(
                (rating_train, (user_train, item_train)), 
                shape=(self.num_users, self.num_items)
            )

            (user_train, user_val,
            item_train, item_val,
            rating_train, rating_val) = train_test_split(
                user_train,
                item_train,
                rating_train,
                test_size=validation_percentage,
                shuffle=True,
                random_state=seed
            )

            self.URM_val = sps.csr_matrix(
                (rating_val, (user_val, item_val)), 
                shape=(self.num_users, self.num_items)
            )

        self.URM_train = sps.csr_matrix(
            (rating_train, (user_train, item_train)), 
            shape=(self.num_users, self.num_items)
        )
            
        self.URM_test = sps.csr_matrix(
            (rating_test, (user_test, item_test)), 
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

    def aggregate_matrixes(self):
        ICM_normalized = sim.normalization.bm25(self.ICM['genre_ICM'])
        aggregated_matrixes_1 = sps.vstack([self.URM_train, ICM_normalized.T])
        aggregated_matrixes_2 = sim.normalization.bm25(sps.vstack([self.URM_train, self.ICM['genre_ICM'].T]))

        return ICM_normalized, aggregated_matrixes_1, aggregated_matrixes_2 

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

    