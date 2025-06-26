"""visulization function for rendered scenes."""

import math
from typing import List, Sequence, Tuple

import matplotlib as mlp
import matplotlib.pyplot as plt


from blockies import blender
from blockies import scene_parameters


def render_grid(
    params: List[scene_parameters.SceneParameters],
    num_cols_per_class: int = 3,
    equal_class_distribution: bool = True,
    label_class_axis: bool = True,
    # see #75 download_blender: bool = False
) -> Tuple[mlp.figure.Figure, Sequence[Sequence[mlp.axes.Axes]]]:
    """Renders scene from a list of SceneParameters and displays the in an image grid.

    Attrs:
        params: List of SceneParameters
        num_cols_per_class: Number of colums of the grid reserved for a single class
        equal_class_distribution: Enforce equal class distribution in visulization.

    """
    healthy_params, ocd_params = scene_parameters.split_healthy_ocd(params)

    if equal_class_distribution:
        number_equal_samples = min(len(healthy_params), len(ocd_params))
        num_rows = int(math.floor(number_equal_samples / num_cols_per_class))
        num_samples_per_class = int(num_rows * num_cols_per_class)
        healthy_params, ocd_params = scene_parameters.split_healthy_ocd(
            params, num_samples_per_class)
    else:
        max_number_samples = max(len(healthy_params), len(ocd_params))
        num_rows = int(math.ceil(max_number_samples / num_cols_per_class))

    middle_column_offset = 1 if label_class_axis else 0
    num_cols = num_cols_per_class * 2 + middle_column_offset

    fig, ax = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(2 * num_cols, 2 * num_rows),
    )

    if label_class_axis:
        ax_title_healthy = ax[0, num_cols_per_class // 2]
        ax_title_healthy.set_title('Healthy', fontsize=20)
        ax_title_ocd = ax[0, num_cols_per_class + num_cols_per_class // 2 + 1]
        ax_title_ocd.set_title('OCD', fontsize=20)

    healthy_ax = ax[:, :num_cols_per_class].flatten().tolist()[::-1]

    # middle_column = ax[:, num_cols_per_class].flatten().tolist()[::-1]
    ocd_ax = ax[:, num_cols_per_class + middle_column_offset:].flatten().tolist()[::-1]

    for (img, mask, param) in blender.render(
            params=healthy_params + ocd_params,
            chunk_size=num_cols_per_class,
            download_blender=True):  # download_blender is true until #75 is fixed
        ax1 = healthy_ax.pop() if param.obj_name == 'healthy' else ocd_ax.pop()
        ax1.axis('off')
        ax1.set_aspect('equal')
        ax1.imshow(img)

    [a.axis('off') for a in ax.flatten()]
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig, ax


def render_single_param(param: scene_parameters.SceneParameters):
    """Renders the image from the given ``SceneParameters`` and plots it.

    Attrs:
        param: SceneParameters of the image
    """
    (img, mask) = blender.render_single(param)
    plt.imshow(img)
    plt.axis('off')
