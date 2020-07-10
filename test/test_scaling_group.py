from unittest.mock import Mock
import forest.scaling_group


def test_scaling_group_scale_up():
    n = 5
    pool = Mock()
    scaling_group = forest.scaling_group.ScalingGroup(pool)
    scaling_group.scale_to(n)
    assert len(scaling_group.instances) == n
    assert pool.acquire.call_count == n


def test_scaling_group_scale_down():
    pool = Mock()
    large = 10
    small = 3
    scaling_group = forest.scaling_group.ScalingGroup(pool)
    scaling_group.scale_to(large)
    scaling_group.scale_to(small)
    assert len(scaling_group.instances) == small
    assert pool.acquire.call_count == large
    assert pool.release.call_count == large - small
