from Recommenders.Base.BaseLightFM import BaseLightFM

class LightFM(BaseLightFM):
    RECOMMENDER_NAME = 'LightFM'

    def __init__(self, URM_train, verbose=True):
        super(LightFM, self).__init__(URM_train, verbose=verbose)
        self.ICM_train = None
        self.UCM_train = None
