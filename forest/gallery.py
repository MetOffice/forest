"""Gallery design pattern"""
from collections import defaultdict
from functools import partial
import forest.state
from forest.reusable_pool import ReusablePool
from forest.scaling_group import ScalingGroup


class Gallery:
    """View orchestration layer"""
    def __init__(self, scaling_groups):
        self.scaling_groups = scaling_groups

    @classmethod
    def profile_view(cls, datasets, figure):
        return cls._from_dataset(datasets, figure, "profile_view")

    @classmethod
    def series_view(cls, datasets, figure):
        return cls._from_dataset(datasets, figure, "series_view")

    @classmethod
    def _from_dataset(cls, datasets, figure, method):
        groups = {}
        for label, dataset in datasets.items():
            if not hasattr(dataset, method):
                continue
            factory = partial(getattr(dataset, method), figure)
            groups[label] = ScalingGroup(ReusablePool(factory))
        return cls(groups)

    def connect(self, store):
        store.add_subscriber(self.render)

    def render(self, state):
        if isinstance(state, dict):
            state = forest.state.State.from_dict(state)

        # Group layers by dataset
        layers = defaultdict(list)
        for uid, settings in sorted(state.layers.index.items()):
            key = settings["dataset"]
            layers[key].append(uid)

        # Apply layer settings to views
        for key, scaling_group in self.scaling_groups.items():
            uids = layers[key]
            scaling_group.scale_to(len(uids))
            for view, uid in zip(scaling_group.instances, uids):
                view.render(state, uid)

        print(layers)
