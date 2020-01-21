"""Driver for Himawari dataset"""


class Dataset:
    """Factory to construct various dataset-specific objects"""
    def navigator(self):
        return Navigator()

    def map_loader(self):
        pass


class Navigator:
    """Dataset wide navigation"""
