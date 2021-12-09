import os
import numpy as np
import pandas as pd
from scipy import sparse as sps
from sklearn.model_selection import train_test_split, StratifiedKFold

class Dataset(object):
    def __init__(self, path='./Data', validation_percentage=0, test_percentage=0.2, k=0, seed=9999):
        self.seed = seed

        # Read and process targets
        targets = self.read_targets_csv(path)
        self.targets = np.unique(targets['user_id'])
        
        # Read and process URM (Note: no need to create new mapping, since all item identifiers are present in the URM)
        # URM stacking not improving results ...
        URM = self.read_URM_csv(path)
        unique_users = URM['user'].unique()
        self.num_users = len(unique_users)
        unique_items = URM['item'].unique()
        self.num_items = len(unique_items)

        if k > 0: 
            self.cross_val = True
            self.k = k
            self.URM_trains, self.URM_tests = self.get_random_kfolds(URM, k, test_percentage)
        else: 
            self.cross_val = False
            self.URM_train, self.URM_val, self.URM_test, self.URM_train_val = self.get_split(
                URM, validation_percentage, test_percentage, seed=self.seed)

        # Read and process ICM's
        ICM_csv = self.read_ICM_csv(path)
        ICM_csv['event_ICM'] = self.cluster_by_episodes(ICM_csv['event_ICM'])

        self.channel_ICM = sps.coo_matrix(
            (ICM_csv['channel_ICM']['data'], (ICM_csv['channel_ICM']['item'], ICM_csv['channel_ICM']['channel']))).tocsr()
        self.event_ICM = sps.coo_matrix(
            (ICM_csv['event_ICM']['data'], (ICM_csv['event_ICM']['item'], ICM_csv['event_ICM']['episodes']))).tocsr()
        self.genre_ICM = sps.coo_matrix(
            (ICM_csv['genre_ICM']['data'], (ICM_csv['genre_ICM']['item'], ICM_csv['genre_ICM']['genre']))).tocsr()
        self.subgenre_ICM = sps.coo_matrix(
            (ICM_csv['subgenre_ICM']['data'], (ICM_csv['subgenre_ICM']['item'], ICM_csv['subgenre_ICM']['subgenre']))).tocsr()
        
        self.ICM = self.get_icm_format_k(11)
    
    # URM
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
    
    def get_df_split(self, col1, col2, col3, split=0.2, seed=None):
        (col1_v1, col1_v2, col2_v1, col2_v2, col3_v1, col3_v2) = train_test_split(
            col1, col2, col3,
            test_size=split,
            random_state=seed
        )

        return (col1_v1, col1_v2, col2_v1, col2_v2, col3_v1, col3_v2)

    def get_csr_from_df(self, col1, col2, col3):
        return sps.coo_matrix(
            (col3, (col1, col2)), 
            shape=(self.num_users, self.num_items)
        ).tocsr()
    
    def get_split(self, data, validation_percentage=0, test_percentage=0.2, seed=None):
        URM_train, URM_val, URM_test, URM_train_val = None, None, None, None

        user_train = data['user']
        item_train = data['item']
        rating_train = data['data']

        if test_percentage > 0: 
            (user_train, user_test,
            item_train, item_test,
            rating_train, rating_test) = self.get_df_split(
                data['user'], data['item'], data['data'], split=test_percentage, seed=seed)

            if validation_percentage > 0:
                URM_train_val = self.get_csr_from_df(user_train, item_train, rating_train)

                (user_train, user_val,
                item_train, item_val,
                rating_train, rating_val) = self.get_df_split(
                    user_train, item_train, rating_train, split=validation_percentage, seed=seed)   

                URM_val = self.get_csr_from_df(user_val, item_val, rating_val)
            URM_test = self.get_csr_from_df(user_test, item_test, rating_test)
        URM_train = self.get_csr_from_df(user_train, item_train, rating_train)

        return URM_train, URM_val, URM_test, URM_train_val

    def get_random_kfolds(self, URM, k, test_percentage):
        URM_trains, URM_tests = [], []
        for _ in range(k):
            self.seed = np.random.randint(0, 10000)
            URM_train, _, URM_test, _ = self.get_split(URM, test_percentage=test_percentage)
            URM_trains.append(URM_train)
            URM_tests.append(URM_test)

        self.seed = 9999
        return URM_trains, URM_tests
    
    def stack_URM_ICM(self, URM_train, ICM):
        stacked_URM = sps.vstack([URM_train, ICM.T])
        stacked_URM = sps.csr_matrix(stacked_URM)

        stacked_ICM = sps.csr_matrix(stacked_URM.T)

        return stacked_URM, stacked_ICM
    
    # ICM
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

        genre_ICM = pd.read_csv(
            os.path.join(path, 'data_ICM_genre.csv'),
            sep=',',
            names=['item', 'genre', 'data'],
            header=0,
            dtype={'item': np.int32, 'col': np.int32, 'data': np.float})

        subgenre_ICM = pd.read_csv(
            os.path.join(path, 'data_ICM_subgenre.csv'),
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

    def cluster_by_episodes(self, df):
        def _make_cluster(x):
            clusters=[3, 10, 30, 100]
            for i, j in enumerate (clusters):
                if x<j: return i
            return len(clusters)
        
        c = df.columns
        df = df.value_counts(subset=c[0]).to_frame('episodes').reset_index()
        df['episodes'] = df['episodes'].apply(_make_cluster)
        df['data'] = 1.
        return df

    def get_icm_format_k(self, k):
        if k == 1: return self.channel_ICM
        if k == 2: return self.event_ICM
        if k == 3: return self.genre_ICM
        if k == 4: return self.subgenre_ICM

        if k == 5: return sps.hstack((self.channel_ICM, self.event_ICM)).tocsr()
        if k == 6: return sps.hstack((self.channel_ICM, self.genre_ICM)).tocsr()
        if k == 7: return sps.hstack((self.channel_ICM, self.subgenre_ICM)).tocsr()
        if k == 8: return sps.hstack((self.event_ICM, self.genre_ICM)).tocsr()
        if k == 9: return sps.hstack((self.event_ICM, self.subgenre_ICM)).tocsr()
        if k == 10: return sps.hstack((self.genre_ICM, self.subgenre_ICM)).tocsr()

        if k == 11: return sps.hstack((self.channel_ICM, self.event_ICM, self.genre_ICM)).tocsr()
        if k == 12: return sps.hstack((self.channel_ICM, self.event_ICM, self.subgenre_ICM)).tocsr()
        if k == 13: return sps.hstack((self.channel_ICM, self.genre_ICM, self.subgenre_ICM)).tocsr()
        if k == 14: return sps.hstack((self.event_ICM, self.genre_ICM, self.subgenre_ICM)).tocsr()
        
        if k == 15: return sps.hstack((self.channel_ICM, self.genre_ICM, self.event_ICM, self.subgenre_ICM)).tocsr()

    # Targets
    def read_targets_csv(self, path):
        targets = pd.read_csv(os.path.join(path, 'data_target_users_test.csv'),
            sep=',',
            names=['user_id'],
            header=0,
            dtype={'user_id': np.int32}
        )

        return targets
        