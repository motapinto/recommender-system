import numpy as np
import scipy.sparse as sps

class _Metrics_Object(object):
    '''
    Abstract class that should be used as superclass of all metrics requiring an object, therefore a state, to be computed
    '''
    def __init__(self):
        pass

    def __str__(self):
        return '{:.4f}'.format(self.get_metric_value())

    def add_recommendations(self, recommended_items_ids):
        raise NotImplementedError()

    def get_metric_value(self):
        raise NotImplementedError()

    def merge_with_other(self, other_metric_object):
        raise NotImplementedError()

class MAP(_Metrics_Object):
    '''
    Mean Average Precision, defined as the mean of the AveragePrecision over all users
    '''

    def __init__(self):
        super(MAP, self).__init__()
        self.cumulative_AP = 0.0
        self.n_users = 0

    def add_recommendations(self, is_relevant, pos_items):
        self.cumulative_AP += average_precision(is_relevant)
        self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_AP/self.n_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is MAP, 'MAP: attempting to merge with a metric object of different type'
        self.cumulative_AP += other_metric_object.cumulative_AP
        self.n_users += other_metric_object.n_users

def average_precision(is_relevant):
    if len(is_relevant) == 0:
        a_p = 0.0
    else:
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float64) / (1 + np.arange(is_relevant.shape[0]))
        a_p = np.sum(p_at_k) / is_relevant.shape[0]

    assert 0 <= a_p <= 1, a_p
    return a_p

class HIT_RATE(_Metrics_Object):
    '''
    Hit Rate, defined as the quota of users that received at least a correct recommendation.
    It is bounded at 1 and is strictly increases as the recommendation list length increases.

    Note that this is different w.r.t. COVERAGE_USER_CORRECT, COVERAGE_USER_CORRECT uses as denominator
    all the users in the dataset, HR only those for which a recommendation was computed.
    In this framework if a user has no possible correct recommendations then it is skipped.
    Therefore HR = COVERAGE_USER_CORRECT / COVERAGE_USER
    '''

    def __init__(self):
        super(HIT_RATE, self).__init__()
        self.cumulative_HR = 0.0
        self.n_users = 0

    def add_recommendations(self, is_relevant):
        self.cumulative_HR += np.any(is_relevant)
        self.n_users += 1

    def get_metric_value(self):
        if self.n_users == 0:
            return 0.0

        return self.cumulative_HR/self.n_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is HIT_RATE, 'HR: attempting to merge with a metric object of different type'
        self.cumulative_HR += other_metric_object.cumulative_HR
        self.n_users += other_metric_object.n_users

def precision(is_relevant):
    if len(is_relevant) == 0:
        precision_score = 0.0
    else:
        precision_score = np.sum(is_relevant, dtype=np.float64) / len(is_relevant)

    assert 0 <= precision_score <= 1, precision_score
    return precision_score

def recall(is_relevant, pos_items):
    recall_score = np.sum(is_relevant, dtype=np.float64) / pos_items.shape[0]
    assert 0 <= recall_score <= 1, recall_score
    return recall_score

class _Global_Item_Distribution_Counter(_Metrics_Object):
    '''
    This abstract class implements the basic functions to calculate the global distribution of items
    recommended by the algorithm and is used by various diversity metrics
    '''
    def __init__(self, n_items, ignore_items):
        super(_Global_Item_Distribution_Counter, self).__init__()

        self.recommended_counter = np.zeros(n_items, dtype=np.float)
        self.ignore_items = ignore_items.astype(np.int).copy()

    def add_recommendations(self, recommended_items_ids):
        if len(recommended_items_ids) > 0:
            self.recommended_counter[recommended_items_ids] += 1

    def _get_recommended_items_counter(self):
        recommended_counter = self.recommended_counter.copy()

        recommended_counter_mask = np.ones_like(recommended_counter, dtype = np.bool)
        recommended_counter_mask[self.ignore_items] = False

        recommended_counter = recommended_counter[recommended_counter_mask]

        return recommended_counter

    def merge_with_other(self, other_metric_object):
        assert isinstance(other_metric_object, self.__class__), '{}: attempting to merge with a metric object of different type'.format(self.__class__)

        self.recommended_counter += other_metric_object.recommended_counter

    def get_metric_value(self):
        raise NotImplementedError()
