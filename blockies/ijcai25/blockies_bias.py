"""This module contains the code to sample ``two4two.SceneParameters``."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Union
import random
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from two4two import utils
from two4two.ijcai25.blockies_scene_parameters import SceneParameters


_Continouos = Union[scipy.stats.rv_continuous, Callable[[], float], float]
Continouos = Union[_Continouos, Dict[str, _Continouos]]

_Discrete = Union[scipy.stats.rv_discrete, Callable[[], float], Callable[[], str], float, str]
Discrete = Union[_Discrete, Dict[str, _Discrete]]

Distribution = Union[Discrete, Continouos]

# Setup ILL characteristics
ILL_MARKERS = {
    1: {'high_bend': .30, 'high_sphere_diff': .30, 'mutation_mainbones': .10, 'stretchy': .30},
    2: {'med_bend': .40, 'med_sphere_diff': .40, 'mutation_color': .20}
}

# # find all possible combinations of chars greather than size 2 and calc PMF
# ILL1_COMBINATIONS = [comb for i in range(2 , len(ILL_MARKERS[1]) + 1) for comb in itertools.combinations(ILL_MARKERS[1], i)]
# ILL1_COMBINATIONS_PMF = {i: 1/len(ILL1_COMBINATIONS) for i in range(len(ILL1_COMBINATIONS))}

# ILL2_COMBINATIONS = [comb for i in range(2 , len(ILL_MARKERS[2]) + 1) for comb in itertools.combinations(ILL_MARKERS[2], i)]
# ILL2_COMBINATIONS_PMF = {i: 1/len(ILL2_COMBINATIONS) for i in range(len(ILL2_COMBINATIONS))}

# HLY_COMBINATIONS = list(itertools.product(ILL_MARKERS[1] + ['']*2, ILL_MARKERS[2] + ['']*2))  # add blank chars for healty with no characteristic changes
# HLY_COMBINATIONS_PMF = {i: 1/len(HLY_COMBINATIONS) for i in range(len(HLY_COMBINATIONS))}

@dataclasses.dataclass()
class Sampler:
    """Samples the parameters of the ``SceneParameters`` objects.

    Attributes describe how the sampling is done. Concretely they provide the color maps for the
    object and the background and the distributors from which the value for the scene parameters are
    drawn.

    Distribution can be: * scipy-distribution from ``scipy.stats`` * callable functions returning a
    single value * a single (default) value. * a dictionary of all before-mentioned types containing
    the keys ``peaky``and ``stretchy``.

    These dictionaries are the easiest way to implement a bias. If you want an attribute to be
    sampled diffrently based on wheter it shows a peaky or stretchy, it is usually sufficient to
    change these dictionaries. See ``ColorBiasedSampler`` as an example.

    To implement more complex biases, you can inherit this class and modify how individual
    attributes are sampled, e.g., by introducing additional dependencies. Usually the best approach
    is to overwrite the sampling method (e.g. ``sample_obj_rotation_pitch``) and modify the sampling
    to be dependent on other attributes. Please be aware that you will then also need to implement
    interventional sampling, because in addition to sampling new parameters, we also want to
    controll an attribute sometimes. That means that we set the attribute to a specific value
    independent of the usual dependencies. If the intervention flag is true, the parameter should be
    sampled independent of any other attribute. For example, if the object color (obj_color)
    depends on the Peaky/Stretchy variable, it would need to be sampled independent
    if intervention = True.

    Since the default sampler implementation in this class is only dependent upon obj_name, so it is
    the only attribute considered in the intervention.

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

    obj_name: Discrete = utils.discrete({'peaky': 1.0, 'stretchy': 0.0})
    ill: Discrete = utils.discrete({0: 1./2., 1: 1./2.})  # define ill classes
    ill_chars: Discrete = dataclasses.field(
        default_factory=lambda: {
            0: utils.discrete({0: .10, 1: .30, 2: .30, 3: .15, 4: .15}),
            1: utils.discrete({2: .75, 3: .20, 4: .05})
        }
    )
    ill_spherical: Continouos = scipy.stats.beta(0.3, 0.3)
    num_diff: Discrete = utils.discrete({1: 1/3, 2: 1/3, 3: 1/3})
    spherical: Continouos = scipy.stats.beta(0.3, 0.3)   
    bending: Continouos = utils.truncated_normal(0.1, 0.125, 0, 0.25)
    arm_position: Continouos = dataclasses.field(
        default_factory=lambda: {
            'peaky': utils.truncated_normal(mean=0.5, std=0.2, lower=0, upper=0.52),
            'stretchy': utils.truncated_normal(mean=0.5, std=0.2, lower=0.48, upper=1.0)
        })
    labeling_error: Discrete = utils.discrete({True: 0., False: 1.})
    obj_rotation_roll: Continouos = utils.truncated_normal(0, 0.03 * np.pi / 4,
                                                           *utils.QUARTER_CIRCLE)
    obj_rotation_pitch: Continouos = utils.truncated_normal(0, 0.3 * np.pi / 4,
                                                            *utils.QUARTER_CIRCLE)
    obj_rotation_yaw: Continouos = scipy.stats.uniform(- np.pi, np.pi)
    fliplr: Discrete = utils.discrete({True: 0., False: 1.})
    position_x: Continouos = scipy.stats.uniform(-0.8, 0.8)
    position_y: Continouos = scipy.stats.uniform(-0.8, 0.8)
    obj_color: Continouos = scipy.stats.uniform(0., 1.)
    bg_color: Continouos = scipy.stats.uniform(0.05, 0.90)
    bg_color_map: str = 'coolwarm'
    obj_color_map: str = 'custum_normal'

    def sample(self, obj_name: Optional[str] = None) -> SceneParameters:
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
        params = SceneParameters()
        self.sample_obj_name(params)
        # The name flag allows to overide the sampling result. The sampling is still executed to
        # trigger any custom functionality that might be implented in subclasses.
        if obj_name and params.obj_name != obj_name:
            params.obj_name = obj_name

        self.sample_ill(params)
        self.sample_ill_chars(params)
        self.sample_arm_position(params)
        self.sample_labeling_error(params)
        self.sample_spherical(params)
        self.sample_ill_spherical(params)
        self.sample_num_diff(params)
        self.sample_bending(params)
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
            return tuple([Sampler._sample(obj_name, dist) for i in range(0, size)])

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

    @staticmethod
    def _sample_truncated(
        obj_name: Optional[str],
        dist: Distribution,
        size: int = 1,
        min: float = float(-np.inf),
        max: float = float(np.inf),
    ) -> Any:
        assert size == 1

        value = Sampler._sample(obj_name, dist, size)
        while not (min <= value <= max):
            value = Sampler._sample(obj_name, dist, size)
        return value
    

    def _sample_name(self) -> str:
        """Convienience function. Returns a sampled obj_name."""
        # obj_name is set to none, because the sampleing of the name should be, per definitiion,
        # idenpendet of the obj_name
        return self._sample(obj_name=None, dist=self.obj_name)

    def sample_obj_name(self, params: SceneParameters):
        """Samples the ``obj_name``."""
        params.obj_name = self._sample(None, self.obj_name)
        params.mark_sampled('obj_name')

    def sample_ill(self, params: SceneParameters, intervention: bool = False) -> str:
        """Sample the ill state        
        
        Attrs:
            params: SceneParameters for which the ill state is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.ill = self._sample(obj_name, self.ill)
        params.mark_sampled('ill')

    def sample_ill_chars(self, params: SceneParameters, intervention: bool = False) -> str:
        """Samples the ill characteristics to use        
        
        Attrs:
            params: SceneParameters for which the ill state is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        ill_state = params.ill
        n_ill_chars = self._sample(ill_state, self.ill_chars)

        if ill_state == 0:
            items = list(ILL_MARKERS[1].keys())
            probs = list(ILL_MARKERS[1].values())
            temp1 = np.random.choice(items, size=1, replace=False, p=probs) # can have up to 1 ill char from variant 1
            temp1 = temp1.tolist()

            items = list(ILL_MARKERS[2].keys())
            probs = list(ILL_MARKERS[2].values())
            temp2 = np.random.choice(items, size=3, replace=False, p=probs) # can have up to 4 ill chars from variant 2 (since not used)
            temp2 = temp2.tolist()

            temp1_chartypes = [c.split('_')[-1] for c in temp1]
            temp2_chartypes = [c.split('_')[-1] for c in temp2]
            remove_idxs = []
            for i, t in enumerate(temp2_chartypes):
                if t in temp1_chartypes:
                    remove_idxs.append(i)
            _ = [temp2.pop(i) for i in sorted(remove_idxs, reverse=True)]

            n_ill_chars = min(len(temp1 + temp2), n_ill_chars)
            params.ill_chars = tuple(np.random.choice(temp1 + temp2, size=n_ill_chars, replace=False).tolist())
        else:
            items = list(ILL_MARKERS[ill_state].keys())
            probs = list(ILL_MARKERS[ill_state].values())
            ill_chars = list(np.random.choice(items, size=n_ill_chars, replace=False, p=probs).tolist())
            ill_chartypes = [c.split('_')[-1] for c in ill_chars]

            # get ill_chars from alternative class and remove chars that are the same attribute
            alt_chars = list(ILL_MARKERS[2].keys())
            alt_types = [c.split('_')[-1] for c in alt_chars]
            remove_idxs = []
            for i, t in enumerate(alt_types):
                if t in ill_chartypes:
                    remove_idxs.append(i)

            _ = [alt_chars.pop(i) for i in sorted(remove_idxs, reverse=True)]

            alt_char_opts = list(ILL_MARKERS[2].keys())
            if 'high_bend' in ill_chars:
                alt_char_opts.remove('med_bend')
            if 'high_sphere_diff' in ill_chars:
                alt_char_opts.remove('med_sphere_diff')

            num_alt = np.random.choice([0,1,2,3], size=1, replace=False, p=[.10, .40, .30, .20])[0]
            num_alt = min(num_alt, len(alt_chars))
            if num_alt > 0:
                alt_char = str(np.random.choice(alt_chars, size=num_alt, replace=False)[0])
                ill_chars.append(alt_char)
            params.ill_chars = tuple(ill_chars)

        params.mark_sampled('ill_chars')

    def sample_arm_position(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``arm_position``.

        Attrs:
            params: SceneParameters for which the arm_position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name

        # check for ill chars for arm position bias
        if 'stretchy' in params.ill_chars:
            obj_name = 'stretchy'

        params.arm_position = float(self._sample(obj_name, self.arm_position))
        params.mark_sampled('arm_position')


    def sample_ill_spherical(self, params: SceneParameters, intervention: bool = False) -> str:
        """Sample the ill state        
        
        Attrs:
            params: SceneParameters for which the ill_spherical value is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name

        dist = utils.truncated_normal(.20, .10, .05, .30)

        assert not('high_sphere_diff' in params.ill_chars and 'med_sphere_diff' in params.ill_chars), "ill_chars must not contain both 'high_sphere_diff' and 'med_sphere_diff'"

        if 'med_sphere_diff' in params.ill_chars:
            dist = utils.truncated_normal(.40, .10, .30, .50)

        if 'high_sphere_diff' in params.ill_chars:
            dist = utils.truncated_normal(.60, .1, .50, .75)

        sphere_diff = -1
        new_val = -1
        while (new_val < 0 or new_val > 1.30):
            sphere_diff = self._sample(obj_name, dist)
            dir = np.random.choice([-1, 1])
            new_val = params.spherical + sphere_diff * dir

        params.ill_spherical = new_val
        params.mark_sampled('ill_spherical')


    def sample_num_diff(self, params: SceneParameters, intervention: bool = False):
        """Sample the the number of componets that should  have different sphericity
        
        Attrs:
            params: SceneParameters for which the num_didff is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name

        params.num_diff = self._sample(obj_name, self.num_diff)
        params.mark_sampled('num_diff')


    def sample_labeling_error(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``labeling_error``.

        Attrs:
            params: SceneParameters for which the labeling_error is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.labeling_error = self._sample(obj_name, self.labeling_error)
        params.mark_sampled('labeling_error')

    def sample_spherical(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``spherical``..

        Attrs:
            params: SceneParameters for which the spherical attribute is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name

        mutation_dist = utils.truncated_normal(1.125, 0.025, 1.11, 1.22)
        dist = self.spherical if 'mutation_mainbones' not in params.ill_chars else mutation_dist
        params.spherical = self._sample(obj_name, dist)

        params.mark_sampled('spherical')

    def sample_bending(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``bending``.

        Attrs:
            params: SceneParameters for which the bone roation is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name

        assert not('high_bend' in params.ill_chars and 'med_bend' in params.ill_chars), "ill_chars must not contain both 'high_bend' and 'med_bend'"

        dist = utils.truncated_normal(0.075, 0.075, 0, 0.13)
        if 'med_bend' in params.ill_chars:
            dist = utils.truncated_normal(0.175, 0.075, 0.13, 0.20)

        if 'high_bend' in params.ill_chars:
            dist = utils.truncated_normal(0.275, 0.05, 0.20, 0.39)

        params.bending = self._sample(obj_name, dist)
        params.mark_sampled('bending')

    def sample_rotation(self, params: SceneParameters, intervention: bool = False):
        """Convienience function bundeling all object rotation functions by calling them.

        Attrs:
            params: SceneParameters for which the object inclination is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        self.sample_obj_rotation_roll(params, intervention=intervention)
        self.sample_obj_rotation_pitch(params, intervention=intervention)
        self.sample_obj_rotation_yaw(params, intervention=intervention)

    def sample_obj_rotation_roll(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``obj_rotation_roll``.

        Attrs:
            params: SceneParameters for which the object inclination is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.obj_rotation_roll = self._sample(obj_name, self.obj_rotation_roll)
        params.mark_sampled('obj_rotation_roll')

    def sample_obj_rotation_pitch(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``obj_rotation_pitch``.

        Attrs:
            params: SceneParameters for which the rotation is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.obj_rotation_pitch = self._sample(obj_name, self.obj_rotation_pitch)
        params.mark_sampled('obj_rotation_pitch')

    def sample_obj_rotation_yaw(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``obj_rotation_yaw``.

        Attrs:
            params: SceneParameters for which the rotation is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.obj_rotation_yaw = self._sample(obj_name, self.obj_rotation_yaw)
        params.mark_sampled('obj_rotation_yaw')

    def sample_fliplr(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``fliplr``.

        Attrs:
            params: SceneParameters for which the fliping (left/right) is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.fliplr = self._sample(obj_name, self.fliplr)
        params.mark_sampled('fliplr')

    def sample_position(self, params: SceneParameters, intervention: bool = False):
        """Convienience function calling ``sample_position_x`` and ``sample_position_y``.

        Attrs:
            params: SceneParameters for which the position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        self.sample_position_x(params, intervention=intervention)
        self.sample_position_y(params, intervention=intervention)

    def sample_position_x(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``position_x`` of the object.

        Attrs:
            params: SceneParameters for which the position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.position_x = self._sample(obj_name, self.position_x)
        params.mark_sampled('position_x')

    def sample_position_y(self, params: SceneParameters, intervention: bool = False):
        """Samples ``position_y`` of the object.

        Attrs:
            params: SceneParameters for which the position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.position_y = self._sample(obj_name, self.position_y)
        params.mark_sampled('position_y')

    def _object_cmap(self, params: SceneParameters) -> mpl.colors.Colormap:
        return plt.get_cmap(self.obj_color_map)

    def sample_color(self, params: SceneParameters, intervention: bool = False):
        """Convienience function calling ``sample_obj_color`` and ``sample_bg_color``.

        Attrs:
            params: SceneParameters for which the position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        self.sample_obj_color(params, intervention=intervention)
        self.sample_bg_color(params, intervention=intervention)

    def sample_obj_color(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``obj_color`` and ``obj_color_rgba``.

        Attrs:
            params: SceneParameters for which the obj_color is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.obj_color = float(self._sample(obj_name, self.obj_color))

        # # cmap = sns.diverging_palette(240, 10, center='dark', as_cmap=True)
        colors = ([0.2564697186166857, 0.4871591358142198, 0.657003695890986, 1.0],
                  [0.15724576976397373, 0.13503847744765152, 0.13576874223703714, 1.0],
                  [0.8545371341681446, 0.22957019267094914, 0.2762321841840895, 1.0])
        cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_diverging", colors)
        if 'mutation_color' in params.ill_chars:
            # cmap = sns.diverging_palette(240, 10, s=100, l=50, center='dark', as_cmap=True)
            colors = ([1.9616380393054912e-05, 0.4953660966080536, 0.7173703514690809, 1.0],
                      [0.16210797334668234, 0.13318563738292316, 0.13418524354237626, 1.0],
                      [0.9347858072201746, 0.0, 0.17084601268703314, 1.0])
            cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_diverging", colors)

        params.obj_color_rgba = tuple(plt.get_cmap(cmap)(params.obj_color))  # type: ignore
        params.mark_sampled('obj_color')

    def _bg_cmap(self, params: SceneParameters) -> mpl.colors.Colormap:
        return plt.get_cmap(self.bg_color_map)

    def sample_bg_color(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``bg_color_rgba`` and ``bg_color``.

        Attrs:
            params: SceneParameters for which the labeling_error is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.bg_color = float(self._sample(obj_name, self.bg_color))
        params.bg_color_rgba = tuple(self._bg_cmap(params)(params.bg_color))  # type: ignore
        params.mark_sampled('bg_color')


@dataclasses.dataclass()
class SimpleColorMapSampler(Sampler):
    """An Sampler with a simpler colormap.

    This colormap allows for very simple experiements with human subjects. The simple colormap makes
    color biases easier to spot.
    """
    bg_color: Continouos = scipy.stats.uniform(0.05, 0.80)
    bg_color_map: str = 'binary'
    obj_color_map: str = 'seismic'


@dataclasses.dataclass()
class ColorBiasedSampler(SimpleColorMapSampler):
    """An example implementation of a color-biased SceneParameterSample.

    The color is sampled from a conditional distribution that is dependent on the object type.
    """

    obj_color: Continouos = dataclasses.field(
        default_factory=lambda: {
            'peaky': utils.truncated_normal(1, 0.5, 0, 1),
            'stretchy': utils.truncated_normal(0, 0.5, 0, 1),
        })


@dataclasses.dataclass()
class HighVariationSampler(Sampler):
    """A sampler producing more challenging images.

    This sampler allows for a higher variation in rotations and bending. Hence it creates a more
    challenging datset.
    """

    obj_rotation_roll: Continouos = scipy.stats.uniform(- np.pi / 3, 2 * np.pi / 3)
    obj_rotation_yaw: Continouos = scipy.stats.uniform(- np.pi, np.pi)
    obj_rotation_pitch: Continouos = scipy.stats.uniform(- np.pi / 3, 2 * np.pi / 3)
    # bending: Continouos = scipy.stats.uniform(- np.pi / 8, np.pi / 4)


@dataclasses.dataclass()
class HighVariationColorBiasedSampler(HighVariationSampler):
    """A sampler producing more challenging images with a color bias that is depent on obj_name.

    This sampler allows for a higher variation in rotations and bending. Hence it creates a more
    challenging datset. This dataset is more challenging. So the bias is more likely to be used.
    """
    obj_color: Continouos = dataclasses.field(
        default_factory=lambda: {
            'peaky': utils.truncated_normal(1, 0.5, 0, 1),
            'stretchy': utils.truncated_normal(0, 0.5, 0, 1),
        })


@dataclasses.dataclass()
class MedVarColorSampler(Sampler):
    """A sampler with a more sophisticated color bias.

    The sample introduces a color bias only for the challenging cases where the arm position is hard
    to distinguish. Therefore, the bias is not evident in every image but informative enough to bias
    a model.
    """

    obj_color: Continouos = dataclasses.field(
        default_factory=lambda: {
            'peaky': utils.truncated_normal(1, 0.5, 0, 1),
            'stretchy': utils.truncated_normal(0, 0.5, 0, 1),
            'peaky_edge': utils.truncated_normal(1, 0.1, 0.7, 1),
            'stretchy_edge': utils.truncated_normal(0, 0.1, 0, 0.3),
        })

    def sample_obj_color(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``obj_color`` and ``obj_color_rgba``.

        Attrs:
            params: SceneParameters for which the obj_color is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        if params.arm_position > 0.45 and params.arm_position < 0.55:
            obj_name = obj_name + "_edge"
        params.obj_color = float(self._sample(obj_name, self.obj_color))
        params.obj_color_rgba = tuple(self._object_cmap(params)(params.obj_color))  # type: ignore
        params.mark_sampled('obj_color')


@dataclasses.dataclass()
class MedVarSampler(Sampler):
    """A sampler with a custom obj_color sampler that has no bias.

    This sampler is the base class for ``MedVarColorSampler'' it uses interventional sampleing to
    avoid introducing a bias.
    """

    def sample_obj_color(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``obj_color`` and ``obj_color_rgba``.

        Attrs:
            params: SceneParameters for which the obj_color is sampled and updated in place.
            intervention: Flag for interventional sampling. False will be ignored for this class.
        """
        # Since the color should be independent of the class we use interventional sampeling.
        super().sample_obj_color(params, intervention=True)


@dataclasses.dataclass()
class MedVarSpherSampler(MedVarSampler):
    """A sampler based on MedVar but with a Spherical bias.

    The sample introduces a spherical bias, but only for the cases that are not challing.
    Since this bias is not informative for all cases another bias can be added that will be used by
    the model if it provides information on the challenging cases.
    """

    def sample_spherical(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``spherical``..

        Attrs:
            params: SceneParameters for which the spherical attribute is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.spherical = self._sample(obj_name, self.spherical)

        if params.arm_position < 0.45:
            params.spherical = self._sample_truncated(
                obj_name, self.spherical, max=0.5)

        if params.arm_position > 0.55:
            params.spherical = self._sample_truncated(
                obj_name, self.spherical, min=0.5)
        params.mark_sampled('spherical')


@dataclasses.dataclass()
class MedVarSpherColorSampler(MedVarColorSampler, MedVarSpherSampler):
    """A sampler that combines the biases of ``MedVarSpherSampler'' and ``MedVarColorSampler''.

    See the the other two classes for documentation.
    """
    pass


@dataclasses.dataclass()
class MedVarNoArmsSampler(MedVarColorSampler):
    """A sampler based on MedVar with spherical and color bias but no arm information.

    This sampler uses interventional sampeling for the arm positon. The sampler is intended to
    produce only validation and test data.
    """
    def sample_arm_position(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``arm_position``.

        Attrs:
            params: SceneParameters for which the arm_position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        super().sample_arm_position(params, intervention=True)



### New Biases for Ill Dataset

@dataclasses.dataclass()
class BendBiasIllSampler(Sampler):
    """Sampler for Sick Ones dataset with a bend biases for the Ill class.
    """

    bending: Continouos = dataclasses.field(
        default_factory=lambda: {
            'peaky_ill_1': utils.truncated_normal(0.2, 0.10, 0.1, 0.35),
            'stretchy_ill_1': utils.truncated_normal(0.2, 0.10, 0.1, 0.35),
            'peaky_ill_2': utils.truncated_normal(0.275, 0.125, 0.125, 0.39),
            'stretchy_ill_2': utils.truncated_normal(0.275,  0.125, 0.125, 0.39),
            'peaky_notill': utils.truncated_normal(0.1, 0.125, 0, 0.25),
            'stretchy_notill': utils.truncated_normal(0.1, 0.125, 0, 0.25),
        })
    
    def sample_bending(self, params: SceneParameters, intervention: bool = False):

        obj_name = self._sample_name() if intervention else params.obj_name
        if params.ill in [1, 2]:
            obj_name = obj_name + f"_ill_{params.ill}"
        else:
            obj_name = obj_name + "_notill"

        bend_abs = float(self._sample(obj_name, self.bending))
        params.bending = bend_abs * np.random.choice([1, -1])
        params.mark_sampled('bending')


@dataclasses.dataclass()
class BendBiasHighVarIllSampler(BendBiasIllSampler):
    """ Base Sampler for Sick Ones dataset with a bend biases for the Ill class, and high variation 
        in rotation to make it slightly more difficult.  
    """
    obj_rotation_roll: Continouos = scipy.stats.uniform(- np.pi / 4, 2 * np.pi / 4)
    obj_rotation_yaw: Continouos = scipy.stats.uniform(- np.pi * 0.75 , np.pi * 0.75)
    obj_rotation_pitch: Continouos = scipy.stats.uniform(- np.pi / 4, 2 * np.pi / 4)


@dataclasses.dataclass()
class BendColorBiasHighVarIllSampler(BendBiasHighVarIllSampler):
    """ Color Biased Sampler for Sick Ones dataset with a bend biases for the Ill class, and high variation 
        in position to make it more difficult.  Plus the color bias that shouldn't be there.
    """

    obj_color: Continouos = dataclasses.field(
        default_factory=lambda: {
            'peaky_ill_1': utils.truncated_normal(1, 0.5, 0, 1),
            'stretchy_ill_1': utils.truncated_normal(1, 0.5, 0, 1),
            'peaky_ill_2': utils.truncated_normal(1, 0.5, 0, 1),
            'stretchy_ill_2': utils.truncated_normal(1, 0.5, 0, 1),
            'peaky_notill': utils.truncated_normal(0, 0.5, 0, 1),
            'stretchy_notill': utils.truncated_normal(0, 0.5, 0, 1),
        })

    def sample_obj_color(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``obj_color`` and ``obj_color_rgba``.

        Attrs:
            params: SceneParameters for which the obj_color is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        if params.ill in [1, 2]:
            obj_name = obj_name + f"_ill_{params.ill}"
        else:
            obj_name = obj_name + "_notill"
        params.obj_color = float(self._sample(obj_name, self.obj_color))
        params.obj_color_rgba = tuple(self._object_cmap(params)(params.obj_color))  # type: ignore
        params.mark_sampled('obj_color')
