"""Tests for ``scene_parameters.py``."""

import dataclasses
import json

import pytest

import blockies


@dataclasses.dataclass
class MyParameters(blockies.SceneParameters):
    """Class to test subclass of SceneParameters."""
    my_field: str = "my very unique value"


def test_printing_scene_parameters():
    """Dummy test of __str__ function."""
    params = blockies.Sampler().sample()
    print(str(params))


def test_subclass_scene_parameters():
    """Tests if load selects the right subclass."""
    param = MyParameters()
    json_buf = json.dumps(param.state_dict())
    loaded_param = blockies.SceneParameters.load(json.loads(json_buf))
    assert param == loaded_param
    assert type(loaded_param) == MyParameters
    assert loaded_param.my_field == "my very unique value"


def test_scene_parameters_loading():
    """Tests if SceneParameters are still equal after loading from json."""
    sampler = blockies.Sampler()
    sampled_param = sampler.sample()
    json_buf = json.dumps(sampled_param.state_dict())
    loaded_param = blockies.SceneParameters.load(json.loads(json_buf))
    assert sampled_param == loaded_param


def test_sample_scene_parameters():
    """Test sampling of SceneParameters."""
    sampler = blockies.Sampler()
    for i in range(1000):
        param = sampler.sample()
        param.check_values()


def test_scene_parameter_clone():
    """Tests the cloning of SceneParameters."""
    param = blockies.SceneParameters()
    param_clone = param.clone()
    assert param_clone != param
    assert param.id != param_clone.id
    assert param_clone.original_id == param.id
    with pytest.raises(ValueError):
        param_clone.clone()
    assert param_clone.clone(create_new_id=False).id == param_clone.id
    assert param_clone.is_cloned()
    assert param_clone.is_clone_of(param)
    assert not param.is_cloned()
