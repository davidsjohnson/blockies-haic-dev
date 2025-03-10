"""This module contains the code to sample ``two4two.SceneParameters``."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from two4two import utils
from two4two.blocky_scene_parameters import BlockySceneParameters


_Continuous = Union[scipy.stats.rv_continuous, Callable[[], float], float]
Continuous = Union[_Continuous, Dict[str, _Continuous]]

_Discrete = Union[scipy.stats.rv_discrete, Callable[[], float], Callable[[], str], float, str, tuple]
Discrete = Union[_Discrete, Dict[str, _Discrete]]

Distribution = Union[Discrete, _Continuous]

# Setup ILL characteristics and their probabilities
ILL_MARKERS = {'high_bend': .30, 'high_sphere_diff': .30, 'mutation_mainbones': .10, 'stretchy': .30}

@dataclasses.dataclass()
class BlockySampler:
    """Samples the parameters of the ``SceneParameters`` objects.

    Attributes describe how the sampling is done. Concretely they provide the color maps for the
    object and the background and the distributors from which the value for the scene parameters are
    drawn.

    Distribution can be: * scipy-distribution from ``scipy.stats`` * callable functions returning a
    single value * a single (default) value. * a dictionary of all before-mentioned types containing
    the keys ``healthy``and ``ocd``.

    These dictionaries are the easiest way to implement a bias. If you want an attribute to be
    sampled diffrently based on wheter it shows a healthy or ocd Blocky, it is usually sufficient to
    change these dictionaries. See ``ColorBiasedSampler`` in the main Two4Two ``bias.py`` as an example.

    To implement more complex biases, you can inherit this class and modify how individual
    attributes are sampled, e.g., by introducing additional dependencies. Usually the best approach
    is to overwrite the sampling method (e.g. ``sample_obj_rotation_pitch``) and modify the sampling
    to be dependent on other attributes. Please be aware that you will then also need to implement
    interventional sampling, because in addition to sampling new parameters, we also want to
    control an attribute sometimes. That means that we set the attribute to a specific value
    independent of the usual dependencies. If the intervention flag is true, the parameter should be
    sampled independent of any other attribute. For example, if the object color (obj_color)
    depends on the Healhty/OCD variable, it would need to be sampled independent
    if intervention = True

    For the valid values ranges, see ``SceneParameters.VALID_VALUES``.

    Attrs:
        bg_color_map: used color map for the background.
        obj_color_map: used color map for the object.
        spherical: distribution of ``SceneParameters.spherical``.
        bending: distribution of ``SceneParameters.bending``.
        obj_name: distribution of ``SceneParameters.obj_name``.
        arm_position: distribution of ``SceneParameters.arm_position_x`` and
            ``SceneParameters.arm_position_y``
        labeling_error: distribution of ``SceneParameters.labeling_error``.
        obj_rotation_roll: distribution of ``SceneParameters.obj_rotation_roll``.
        obj_rotation_pitch:distribution of ``SceneParameters.obj_rotation_pitch``.
        obj_rotation_yaw:distribution of ``SceneParameters.obj_rotation_pitch``.
        fliplr: distribution of ``SceneParameters.fliplr``.
        position: distribution of ``SceneParameters.position``.
        obj_color: distribution of ``SceneParameters.obj_color``.
        bg_color: distribution of ``SceneParameters.bg_color``.
    """

    # set the default sampling distributions
    obj_name: Discrete = {'healthy': 1.0, 'ocd': 0.0}
    num_ill_chars: Discrete = dataclasses.field(
        default_factory=lambda: {
            'healthy': utils.discrete({0: .10, 1: .30, 2: .30, 3: .15, 4: .15}),
            'ocd': utils.discrete({2: .75, 3: .20, 4: .05})
        }
    )
    # by default choose 1 ill charachteristic for healthy and 2 for ocd
    ill_chars: Discrete = dataclasses.field(
        default_factory=lambda: {
            'healthy': utils.multiple_choice(values=list(ILL_MARKERS.keys()), probs=list(ILL_MARKERS.values()), size=1),
            'ocd': utils.multiple_choice(values=list(ILL_MARKERS.keys()), probs=list(ILL_MARKERS.values()), size=2)
        }
    )

    # set up main and secondary bones (sec_spherical needs to be updated in sampler function since it is depen)
    main_spherical: Continuous = scipy.stats.beta(0.3, 0.3)
    sec_spherical: Continuous = utils.truncated_normal(.20, .10, .05, .30)
    num_sec_bones: Discrete = utils.discrete({1: 1/3, 2: 1/3, 3: 1/3})

    bending: Continuous = utils.truncated_normal(0.1, 0.125, 0, 0.20)
    arm_position: Continuous = utils.truncated_normal(mean=0.5, std=0.2, lower=0, upper=0.5)
    labeling_error: Discrete = utils.discrete({True: 0., False: 1.})
    obj_rotation_roll: Continuous = utils.truncated_normal(0, 0.03 * np.pi / 4,
                                                           *utils.QUARTER_CIRCLE)
    obj_rotation_pitch: Continuous = utils.truncated_normal(0, 0.3 * np.pi / 4,
                                                            *utils.QUARTER_CIRCLE)
    obj_rotation_yaw: Continuous = scipy.stats.uniform(- np.pi, np.pi)
    fliplr: Discrete = utils.discrete({True: 0., False: 1.})
    position_x: Continuous = scipy.stats.uniform(-0.8, 0.8)
    position_y: Continuous = scipy.stats.uniform(-0.8, 0.8)
    obj_color: Continuous = scipy.stats.uniform(0., 1.)
    bg_color: Continuous = scipy.stats.uniform(0.05, 0.90)
    bg_color_map: str = 'coolwarm'
    obj_color_map: str = 'custum_normal'

    
    def sample(self, obj_name: Optional[str] = None) -> BlockySceneParameters:
        """Returns a new SceneParameters with random values.

        If you create your own biased sampled dataset by inheriting from this class,
        you might want to change the order of how attributes are set.
        For example, if you want that ``obj_rotation_pitch`` should depend on the
        ``arm_position``then you should also sample the ``arm_position`` first.
        However, it is highly recommended to sample the object name first, as
        the sampling of the attribute might be dependent on the label
        (see the explanation of distributions in class description)

        Attrs:
            obj_name: Overides the sampled obj_name with the given namen. Usally only useful for
                manual sampling. Not recommeded when samplign larger sets.
        """
        params = BlockySceneParameters()
        self.sample_obj_name(params)
        # The name flag allows to overide the sampling result. The sampling is still executed to
        # trigger any custom functionality that might be implented in subclasses.
        if obj_name and params.obj_name != obj_name:
            params.obj_name = obj_name

        self.sample_num_ill_chars(params)
        self.sample_ill_chars(params)
        self.sample_labeling_error(params)  
        self.sample_main_spherical(params)
        self.sample_sec_spherical(params)
        self.sample_num_sec_bones(params)
        self.sample_bending(params)
        self.sample_arm_position(params)
        self.sample_rotation(params)
        self.sample_fliplr(params)
        self.sample_position(params)
        self.sample_color(params)
        params.check_values()
        return params
    
    @staticmethod
    def _sample(obj_name: Optional[str], dist: Distribution, size: int = 1) -> Any:
        """Samples values from the distributon according to its type.

        The default number of values sampled is one, which can be changed with flag size.

        Distribution can be:
        * scipy-distribution from ``scipy.stats``
        * callable functions returning a single value
        * a single (default) value.
        * a dictionary of all before-mentioned types containing the keys ``peaky``and ``stretchy``.

        Will unpack np.ndarray, list, or tuple with a single element returned by distribution.

        """

        if size > 1:
            return tuple([BlockySampler._sample(obj_name, dist) for i in range(0, size)])

        if isinstance(dist, dict):
            # Rare edge case: If a dictionary was passed without the obj_name key
            # then the first distribution from the dictionary is used.
            if obj_name is None:
                dist = next(iter(dist.values()))
            else:
                dist = dist[obj_name]

        if hasattr(dist, 'rvs'):
            value = dist.rvs()  # type: ignore
        elif callable(dist):
            value = dist()
        else:
            value = dist

        # Unpacking float values contained in numpyarrays and list
        if type(value) in (list, tuple):
            if len(value) != 1:
                raise ValueError(f"Expected a single element. \
                 Got {type(value)} of size {len(value)}!")
            else:
                value = value[0]

        if isinstance(value, np.ndarray):
            value = utils.numpy_to_python_scalar(value)

        return value

    
    def sample_obj_name(self, params: BlockySceneParameters):
        """Samples the ``obj_name``."""
        params.obj_name = self._sample(None, self.obj_name)
        params.mark_sampled('obj_name')

    def sample_labeling_error(self, params: BlockySceneParameters, intervention: bool = False):
        """Samples the ``labeling_error``.

        Attrs:
            params: SceneParameters for which the labeling_error is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.labeling_error = self._sample(obj_name, self.labeling_error)
        params.mark_sampled('labeling_error')

    def sample_num_ill_chars(self, params: BlockySceneParameters):
        """Samples the number of ill characteristics to use for Blocky sampling disributions."""
        params.num_ill_chars = self._sample(params.obj_name, self.num_ill_chars)
        params.mark_sampled('num_ill_chars')

    def sample_ill_chars(self, params: BlockySceneParameters, intervention: bool = False) -> str:
        """Samples the ill characteristics to use for Blocky sampling disributions.
        
        Attrs:
            params: SceneParameters for which the ill state is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        num_ill_chars = params.num_ill_chars

        assert num_ill_chars < 2 if obj_name == 'healthy' else num_ill_chars <= 4, f'Invalid number of ill characteristics, {num_ill_chars} for the given obj_name {obj_name}'

        char_sampler = utils.multiple_choice(values=list(ILL_MARKERS.keys()), probs=list(ILL_MARKERS.values()), size=num_ill_chars)
        params.ill_chars = char_sampler()

        params.mark_sampled('ill_chars')

    # todo: research how to decorate function with trait based distributions for more flexibility
    def sample_main_spherical(self, params: BlockySceneParameters, intervention: bool = False):
        """Samples the shape of the main bones.

        Attrs:
            params: SceneParameters for which the ill state is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name

        mutation_dist = utils.truncated_normal(1.125, 0.025, 1.11, 1.22)
        dist = mutation_dist if params.ill_chars == 'mutation_mainbones' else self.main_spherical
        params.main_spherical = self._sample(obj_name, dist)
        params.mark_sampled('main_spherical')

    # todo: how to better handle this function
    def sample_sec_spherical(self, params: BlockySceneParameters, intervention: bool = False):
        """Samples the shape of the secondary bones.

        Attrs:
            params: SceneParameters for which the ill state is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name

        high_sphere_dist = utils.truncated_normal(.60, .1, .50, .75)

        dist = high_sphere_dist if params.ill_chars == 'high_sphere_diff' else self.sec_spherical

        sphere_diff = -1
        new_val = -1
        while (new_val < 0 or new_val > 1.30):
            sphere_diff = self._sample(obj_name, dist)
            dir = np.random.choice([-1, 1])
            new_val = params.main_spherical + sphere_diff * dir

        params.sec_spherical = new_val
        params.mark_sampled('sec_spherical')

    def sample_num_sec_bones(self, params: BlockySceneParameters, intervention: bool = False):
        """Samples the number of secondary bones.

        Attrs:
            params: SceneParameters for which the ill state is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.num_sec_bones = self._sample(obj_name, self.num_sec_bones)
        params.mark_sampled('num_sec_bones')

    def sample_bending(self, params: BlockySceneParameters, intervention: bool = False):
        """Samples the bending of the bones.

        Attrs:
            params: SceneParameters for which the ill state is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """

        obj_name = self._sample_name() if intervention else params.obj_name

        high_bend_dist = utils.truncated_normal(0.275, 0.05, 0.20, 0.39)

        dist = high_bend_dist if 'high_bend' in params.ill_chars else self.bending
        params.bending = self._sample(obj_name, dist)
        params.mark_sampled('bending')

    def sample_arm_position(self, params: BlockySceneParameters, intervention: bool = False):
        """Samples the arm position.

        Attrs:
            params: SceneParameters for which the ill state is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name

        stretchy_dist = utils.truncated_normal(mean=0.75, std=0.1, lower=0.5, upper=1.0)
        dist = stretchy_dist if 'stretchy' in params.ill_chars else self.arm_position
        params.arm_position = self._sample(obj_name, dist)
        params.mark_sampled('arm_position')


    def sample_rotation(self, params: BlockySceneParameters, intervention: bool = False):
        """Convienience function bundeling all object rotation functions by calling them.

        Attrs:
            params: SceneParameters for which the object inclination is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        self.sample_obj_rotation_roll(params, intervention=intervention)
        self.sample_obj_rotation_pitch(params, intervention=intervention)
        self.sample_obj_rotation_yaw(params, intervention=intervention)

    def sample_obj_rotation_roll(self, params: BlockySceneParameters, intervention: bool = False):
        """Samples the ``obj_rotation_roll``.

        Attrs:
            params: SceneParameters for which the object inclination is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.obj_rotation_roll = self._sample(obj_name, self.obj_rotation_roll)
        params.mark_sampled('obj_rotation_roll')

    def sample_obj_rotation_pitch(self, params: BlockySceneParameters, intervention: bool = False):
        """Samples the ``obj_rotation_pitch``.

        Attrs:
            params: SceneParameters for which the rotation is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.obj_rotation_pitch = self._sample(obj_name, self.obj_rotation_pitch)
        params.mark_sampled('obj_rotation_pitch')

    def sample_obj_rotation_yaw(self, params: BlockySceneParameters, intervention: bool = False):
        """Samples the ``obj_rotation_yaw``.

        Attrs:
            params: SceneParameters for which the rotation is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.obj_rotation_yaw = self._sample(obj_name, self.obj_rotation_yaw)
        params.mark_sampled('obj_rotation_yaw')

    def sample_fliplr(self, params: BlockySceneParameters, intervention: bool = False):
        """Samples the ``fliplr``.

        Attrs:
            params: SceneParameters for which the fliping (left/right) is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.fliplr = self._sample(obj_name, self.fliplr)
        params.mark_sampled('fliplr')

    def sample_position(self, params: BlockySceneParameters, intervention: bool = False):
        """Convienience function calling ``sample_position_x`` and ``sample_position_y``.

        Attrs:
            params: SceneParameters for which the position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        self.sample_position_x(params, intervention=intervention)
        self.sample_position_y(params, intervention=intervention)

    def sample_position_x(self, params: BlockySceneParameters, intervention: bool = False):
        """Samples the ``position_x`` of the object.

        Attrs:
            params: SceneParameters for which the position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.position_x = self._sample(obj_name, self.position_x)
        params.mark_sampled('position_x')

    def sample_position_y(self, params: BlockySceneParameters, intervention: bool = False):
        """Samples ``position_y`` of the object.

        Attrs:
            params: SceneParameters for which the position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.position_y = self._sample(obj_name, self.position_y)
        params.mark_sampled('position_y')

    def _object_cmap(self, params: BlockySceneParameters) -> mpl.colors.Colormap:
        return plt.get_cmap(self.obj_color_map)

    def sample_color(self, params: BlockySceneParameters, intervention: bool = False):
        """Convienience function calling ``sample_obj_color`` and ``sample_bg_color``.

        Attrs:
            params: SceneParameters for which the position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        self.sample_obj_color(params, intervention=intervention)
        self.sample_bg_color(params, intervention=intervention)

    def sample_obj_color(self, params: BlockySceneParameters, intervention: bool = False):
        """Samples the ``obj_color`` and ``obj_color_rgba``.

        Attrs:
            params: SceneParameters for which the obj_color is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.obj_color = float(self._sample(obj_name, self.obj_color))
        params.obj_color_rgba = tuple(self._object_cmap(params)(params.obj_color))  # type: ignore
        params.mark_sampled('obj_color')

    def _bg_cmap(self, params: BlockySceneParameters) -> mpl.colors.Colormap:
        return plt.get_cmap(self.bg_color_map)

    def sample_bg_color(self, params: BlockySceneParameters, intervention: bool = False):
        """Samples the ``bg_color_rgba`` and ``bg_color``.

        Attrs:
            params: SceneParameters for which the labeling_error is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.bg_color = float(self._sample(obj_name, self.bg_color))
        params.bg_color_rgba = tuple(self._bg_cmap(params)(params.bg_color))  # type: ignore
        params.mark_sampled('bg_color')