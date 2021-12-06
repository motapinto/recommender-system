import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import similaripy as sim
from scipy import sparse as sps
from sklearn.model_selection import train_test_split, StratifiedKFold

class Dataset(object):
    def __init__(self, path='./Data', validation_percentage=0.1, test_percentage=0.1):
        self.seed = 9999

        # Read and process targets
        targets = self.read_targets_csv(path)
        self.targets = np.unique(targets['user_id'])
        
        # Read and process URM (Note: no need to create new mapping, since all item identifiers are present in the URM)
        URM = self.read_URM_csv(path)
        unique_users = URM['user'].unique()
        self.num_users = len(unique_users)
        unique_items = URM['item'].unique()
        self.num_items = len(unique_items)

        # AUX
        URM = self.URM_to_csr(URM)
        self.URM_train, self.URM_test = self.train_test_holdout_adjusted(URM, 0.9)
        self.URM_train_2, self.URM_test_2 = self.train_test_holdout_adjusted_col(URM, 0.9)

        # Read and process ICM's
        ICM = self.read_ICM_csv_csv(path)
        self.channel_ICM = sps.coo_matrix(
            (ICM['channel_ICM']['data'], (ICM['channel_ICM']['item'], ICM['channel_ICM']['channel']))).tocsr()
        self.event_ICM = sps.coo_matrix(
            (ICM['event_ICM']['data'], (ICM['event_ICM']['item'], ICM['event_ICM']['episode']))).tocsr()
        self.genre_ICM = sps.coo_matrix(
            (ICM['genre_ICM']['data'], (ICM['genre_ICM']['item'], ICM['genre_ICM']['genre']))).tocsr()
        self.subgenre_ICM = sps.coo_matrix(
            (ICM['subgenre_ICM']['data'], (ICM['subgenre_ICM']['item'], ICM['subgenre_ICM']['subgenre']))).tocsr()
        
        self.ICM = sps.hstack((self.channel_ICM, self.genre_ICM)).tocsr() # self.subgenre_ICM, self.event_ICM
    
    def read_URM_csv(self, path):
        URM = pd.read_csv(os.path.join(path, 'data_train.csv'),
            sep=',',
            names=['user', 'item', 'data'],
            header=0,
            dtype={'row': np.int32, 'col': np.int32, 'data': np.int32}
        )

        return URM
    
    def URM_to_csr(self, URM):
        user_list = np.asarray(list(URM.user))
        item_list_urm = np.asarray(list(URM.item))
        interaction_list = list(np.ones(len(item_list_urm)))
        URM = sps.coo_matrix((interaction_list, (user_list, item_list_urm)), dtype=np.float64)
        return URM.tocsr()
    
    def get_split(self, data, validation_percentage=0.1, test_percentage=0.1):
        seed = 9999
        URM_train, URM_val, URM_test = None, None, None

        (user_train, user_test,
        item_train, item_test,
        rating_train, rating_test) = train_test_split(
            data['user'],
            data['item'],
            data['data'],
            test_size=test_percentage,
            random_state=seed
        )

        if validation_percentage > 0:
            self.URM_train_val = sps.coo_matrix(
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

            URM_val = sps.coo_matrix(
                (rating_val, (user_val, item_val)), 
                shape=(self.num_users, self.num_items)
            ).tocsr()

        URM_train = sps.coo_matrix(
            (rating_train, (user_train, item_train)), 
            shape=(self.num_users, self.num_items)
        ).tocsr()
            
        URM_test = sps.coo_matrix(
            (rating_test, (user_test, item_test)), 
            shape=(self.num_users, self.num_items)
        ).tocsr()

        return URM_train, URM_val, URM_test

    def read_ICM_csv(self, path):
        channel_ICM = pd.read_csv(
            os.path.join(path, 'data_ICM_channel.csv'),
            sep=',',
            names=['item', 'channel', 'data'],
            header=0,
            dtype={'item': np.int32, 'col': np.int32, 'data': np.float})
        
        event_ICM = pd.read_csv(
            os.path.join(path, 'data_ICM_event.csv'),
            sep=',',
            names=['item', 'episode', 'data'],
            header=0,
            dtype={'item': np.int32, 'col': np.int32, 'data': np.float})
        # print(event_ICM.head())
        # event_ICM = event_ICM.groupby('item').count()
        # print(event_ICM.head())

        genre_ICM = pd.read_csv(
            os.path.join(path, 'data_ICM_genre.csv'),
            sep=',',
            names=['item', 'genre', 'data'],
            header=0,
            dtype={'item': np.int32, 'col': np.int32, 'data': np.float})

        subgenre_ICM = pd.read_csv(
            os.path.join(path, 'data_ICM_genre.csv'),
            sep=',',
            names=['item', 'subgenre', 'data'],
            header=0,
            dtype={'item': np.int32, 'col': np.int32, 'data': np.float})

        ICM = { 
            'channel_ICM': channel_ICM, 
            'event_ICM': event_ICM, 
            'genre_ICM': genre_ICM, 
            'subgenre_ICM': subgenre_ICM
        }

        return ICM

    def read_targets_csv(self, path):
        targets = pd.read_csv(os.path.join(path, 'data_target_users_test.csv'),
            sep=',',
            names=['user_id'],
            header=0,
            dtype={'user_id': np.int32}
        )

        return targets

    def train_test_holdout_adjusted(self, URM_all, train_perc = 0.8):
        URM_all = URM_all.tocoo()

        temp_col_num = 0
        train_mask = np.array([]).astype(bool)
        prev_row = URM_all.row[0]

        for k in tqdm(range(len(URM_all.row))):

            if URM_all.row[k] == prev_row:
                temp_col_num += 1
            else:
                if temp_col_num >= 10:
                    temp_mask = np.random.choice([True, False], temp_col_num, p=[train_perc, 1 - train_perc])
                else:
                    temp_mask = np.repeat(True, temp_col_num)

                train_mask = np.append(train_mask, temp_mask)
                temp_col_num = 1

            if k == len(URM_all.row)-1:
                temp_mask = np.random.choice([True, False], temp_col_num, p=[train_perc, 1 - train_perc])
                train_mask = np.append(train_mask, temp_mask)

            prev_row = URM_all.row[k]


        URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])))
        URM_train = URM_train.tocsr()

        test_mask = np.logical_not(train_mask)

        URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])))
        URM_test = URM_all.tocsr()

        return URM_train, URM_test

    def train_test_holdout_adjusted_col(self, URM_all, train_perc = 0.8):
        URM_all = URM_all.T.tocoo()

        temp_col_num = 0
        train_mask = np.array([]).astype(bool)
        prev_row = URM_all.row[0]

        for k in tqdm(range(len(URM_all.row))):

            if URM_all.row[k] == prev_row:
                temp_col_num += 1
            else:
                if temp_col_num >= 10:
                    temp_mask = np.random.choice([True, False], temp_col_num, p=[train_perc, 1 - train_perc])
                else:
                    temp_mask = np.repeat(True, temp_col_num)

                train_mask = np.append(train_mask, temp_mask)
                temp_col_num = 1

            if k == len(URM_all.row)-1:
                temp_mask = np.random.choice([True, False], temp_col_num, p=[train_perc, 1 - train_perc])
                train_mask = np.append(train_mask, temp_mask)

            prev_row = URM_all.row[k]


        URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])))
        URM_train = URM_train.T.tocsc()

        test_mask = np.logical_not(train_mask)

        URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])))
        URM_test = URM_all.T.tocsc()

        return URM_train, URM_test

        