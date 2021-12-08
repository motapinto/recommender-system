from Recommenders.Base.BaseLightFM import BaseLightFM

class LightFMCF(BaseLightFM):
    RECOMMENDER_NAME = 'LightFMCF'

    def __init__(self, URM_train, verbose=True):
        super(LightFMCF, self).__init__(URM_train, verbose=verbose)
        self.ICM_train = None
        self.UCM_train = None
