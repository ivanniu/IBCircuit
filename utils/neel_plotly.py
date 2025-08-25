# code source: https://github.com/neelnanda-io/neel-plotly/tree/main
import torch
import numpy as np
import einops
import plotly.express as px
from functools import partial
import copy
import re
import plotly.graph_objects as go
import pandas as pd
import os

DEFAULT_KWARGS = dict(
    xaxis="x",  # Good
    yaxis="y",  # Good
    range_x=None,  # Good
    range_y=None,  # Good
    animation_name="snapshot",  # Good
    color_name="Color",  # Good
    color=None,
    log_x=False,  # Good
    log_y=False,  # Good
    toggle_x=False,  # Good
    toggle_y=False,  # Good
    legend=True,  # Good
    hover=None,  # Good
    hover_name="data",  # GOod
    return_fig=False,  # Good
    animation_index=None,  # Good
    line_labels=None,  # Good
    markers=False,  # Good
    frame_rate=None,  # Good
    facet_labels=None,
    facet_name="facet",
    include_diag=False,
    debug=False,
    transition="none",  # If "none" then turns off animation transitions, it just jumps between frames
    animation_maxrange_x=True,  # Figure out the maximal range if animation across all frames and fix
    animation_maxrange_y=True,  # Figure out the maximal range if animation across all frames and fix
)


def split_kwargs(kwargs):
    custom = dict(DEFAULT_KWARGS)
    plotly = {}
    for k, v in kwargs.items():
        if k in custom:
            custom[k] = v
        else:
            plotly[k] = v
    return custom, plotly


def to_numpy_ragged_2d(lists):
    # Assumes input is a ragged list (of lists, tensors or arrays). Further assumes it's 2D
    lists = list(map(to_numpy, lists))
    a = len(lists)
    b = max(map(len, lists))
    base_array = np.ones((a, b))
    base_array.fill(np.NINF)
    for i in range(a):
        base_array[i, : len(lists[i])] = lists[i]
    return base_array


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        # if isinstance(tensor[0])
        tensor = list(map(to_numpy, tensor))
        array = np.array(tensor)
        if array.dtype != np.dtype("O"):
            return array
        else:
            return to_numpy_ragged_2d(tensor)
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.detach().cpu().numpy()
    elif type(tensor) in [int, float, bool, str]:
        return np.array(tensor)
    elif isinstance(tensor, pd.Series):
        return tensor.values
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


def update_hovertemplate(data, string):
    if data.hovertemplate is not None:
        data.hovertemplate = (
                data.hovertemplate[:-15] + "<br>" + string + "<extra></extra>"
        )


def update_data(data, custom_kwargs, index):
    if custom_kwargs["hover"] is not None and isinstance(data, go.Heatmap):
        # Assumption -
        hover = custom_kwargs["hover"]
        hover_name = custom_kwargs["hover_name"]
        hover = to_numpy(hover)
        data.customdata = hover
        update_hovertemplate(data, f"{hover_name}=%{{customdata}}")
    if custom_kwargs["markers"]:
        data["mode"] = "lines+markers"
    if custom_kwargs["line_labels"] is not None:
        data["name"] = custom_kwargs["line_labels"][index]
        data["hovertemplate"] = re.sub(
            f"={index}", f"={data['name']}", data["hovertemplate"]
        )
    return


def update_frame(frame, custom_kwargs, frame_index):
    # if custom_kwargs['animation_index'] is not None:
    #     frame['name'] = custom_kwargs['animation_index'][frame_index]
    update_data_list(frame["data"], custom_kwargs)
    return


def add_button(layout, button, pos=None):
    if pos is None:
        num_prev_buttons = len(layout.updatemenus)
        button["y"] = 1 - num_prev_buttons * 0.15
    else:
        button["y"] = pos
    if "x" not in button:
        button["x"] = -0.1
    layout.updatemenus = layout.updatemenus + (button,)


def add_axis_toggle(layout, axis, pos=None):
    assert axis in "xy", f"Invalid axis: {axis}"
    is_already_log = layout[f"{axis}axis"].type == "log"
    toggle_axis = dict(
        type="buttons",
        active=0 if is_already_log else -1,
        buttons=[
            dict(
                label=f"Log {axis}-axis",
                method="relayout",
                args=[{f"{axis}axis.type": "log"}],
                args2=[{f"{axis}axis.type": "linear"}],
            )
        ],
    )
    add_button(layout, toggle_axis, pos=pos)


def update_play_button(button, custom_kwargs):
    if custom_kwargs["transition"] == "none":
        button.args[1]["transition"]["duration"] = 0
    else:
        button.args[1]["transition"]["easing"] = custom_kwargs["transition"]
    if custom_kwargs["frame_rate"] is not None:
        button.args[1]["frame"]["duration"] = custom_kwargs["frame_rate"]


def update_layout(layout, custom_kwargs, is_animation):
    if custom_kwargs["debug"]:
        print(layout, is_animation)
    layout.xaxis.title.text = custom_kwargs["xaxis"]
    layout.yaxis.title.text = custom_kwargs["yaxis"]
    if custom_kwargs["log_x"]:
        layout.xaxis.type = "log"
        if custom_kwargs["range_x"] is not None:
            range_x_0, range_x_1 = custom_kwargs["range_x"]
            layout.xaxis.range = (np.log10(range_x_0), np.log10(range_x_1))
    else:
        if custom_kwargs["range_x"] is not None:
            layout.xaxis.range = custom_kwargs["range_x"]
    if custom_kwargs["log_y"]:
        layout.yaxis.type = "log"
        if custom_kwargs["range_y"] is not None:
            range_y_0, range_y_1 = custom_kwargs["range_y"]
            layout.yaxis.range = (np.log10(range_y_0), np.log10(range_y_1))
    else:
        if custom_kwargs["range_y"] is not None:
            layout.yaxis.range = custom_kwargs["range_y"]
    if custom_kwargs["toggle_x"]:
        add_axis_toggle(layout, "x")
    if custom_kwargs["toggle_y"]:
        add_axis_toggle(layout, "y")
    if not custom_kwargs["legend"]:
        layout.showlegend = False
    if custom_kwargs["facet_labels"]:
        for i, label in enumerate(custom_kwargs["facet_labels"]):
            layout.annotations[i]["text"] = label
            if i > 0:
                layout[f"xaxis{i + 1}"].title = layout["xaxis"].title

    if is_animation:
        for updatemenu in layout.updatemenus:
            if "buttons" in updatemenu:
                for button in updatemenu["buttons"]:
                    if button.label == "&#9654;":
                        update_play_button(button, custom_kwargs)
        layout.sliders[0].currentvalue.prefix = custom_kwargs["animation_name"] + "="
        if custom_kwargs["animation_index"] is not None:
            steps = layout.sliders[0].steps
            for c, step in enumerate(steps):
                step.label = custom_kwargs["animation_index"][c]


def update_data_list(data_list, custom_kwargs):
    for c, data in enumerate(data_list):
        update_data(data, custom_kwargs, c)
    return


def update_fig(fig, custom_kwargs, inplace=True):
    if custom_kwargs["debug"]:
        print(fig.frames == tuple())
    if not inplace:
        fig = copy.deepcopy(fig)
    update_data_list(fig["data"], custom_kwargs)
    is_animation = "frames" in fig and fig.frames != tuple()
    if is_animation:
        for frame_index, frame in enumerate(fig["frames"]):
            update_frame(frame, custom_kwargs, frame_index)
    update_layout(fig.layout, custom_kwargs, is_animation)
    return fig


def imshow_base(array, filename, plot_dir, **kwargs):
    array = to_numpy(array)
    custom_kwargs, plotly_kwargs = split_kwargs(kwargs)
    array = to_numpy(array)
    fig = px.imshow(array, **plotly_kwargs)
    update_fig(fig, custom_kwargs)
    if custom_kwargs["return_fig"]:
        return fig
    # else:
    #     fig.show()
    fig.write_html(plot_dir + filename + ".html")
    fig.write_image(plot_dir + filename + ".png")


imshow = partial(
    imshow_base,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
    aspect="auto",
)


def broadcast_up(array, shape, axis_str=None):
    n = len(shape)
    m = len(array.shape)
    if axis_str is None:
        axis_str = " ".join([f"x{i}" for i in range(n - m, n)])
    return einops.repeat(
        array,
        f"{axis_str}->({' '.join([f'x{i}' for i in range(n)])})",
        **{f"x{i}": shape[i] for i in range(n)},
    )


def melt(tensor):
    arr = to_numpy(tensor)
    n = arr.ndim
    grid = np.ogrid[tuple(map(slice, arr.shape))]
    out = np.empty(arr.shape + (n + 1,), dtype=np.result_type(arr.dtype, int))
    offset = 1

    for i in range(n):
        out[..., i + offset] = grid[i]
    out[..., -1 + offset] = arr
    out.shape = (-1, n + 1)

    df = pd.DataFrame(out, columns=["value"] + [str(i) for i in range(n)], dtype=float)
    return df.convert_dtypes([float] + [int] * n)


def line_or_scatter(
        y, filename, x=None, mode="multi", squeeze=True, plot_type=None, animation_frame=None, facet_col=None, **kwargs
):
    custom_kwargs, plotly_kwargs = split_kwargs(kwargs)
    array = to_numpy(y)
    animation_name = custom_kwargs["animation_name"]
    xaxis = custom_kwargs["xaxis"]
    yaxis = custom_kwargs["yaxis"]
    color_name = custom_kwargs["color_name"]
    color = custom_kwargs["color"]
    facet_name = custom_kwargs["facet_name"]
    if custom_kwargs["debug"]:
        print(color, color_name)
    if squeeze:
        array = array.squeeze()

    df = melt(array)

    _color_name = None
    _animation_name = None
    _facet_col = None
    if plot_type == "line":
        # x and y
        cols_not_taken = list(range(len(df.columns) - 1))
        if animation_frame is not None:
            _animation_name = str(animation_frame)

            cols_not_taken.remove(animation_frame)

        if facet_col is not None:
            _facet_col = str(facet_col)

            cols_not_taken.remove(facet_col)

        if len(cols_not_taken) == 1:
            _color_name = None
            _x_name = str(cols_not_taken[0])
        elif len(cols_not_taken) == 2:
            _color_name = str(cols_not_taken[0])
            _x_name = str(cols_not_taken[1])
        else:
            raise ValueError(f"Input tensor has too many dimensions: {array.shape}. Available cols {cols_not_taken}")

    else:
        cols_not_taken = list(range(len(df.columns) - 1))
        if animation_frame is not None:
            _animation_name = str(animation_frame)

            cols_not_taken.remove(animation_frame)

        if facet_col is not None:
            _facet_col = str(facet_col)

            cols_not_taken.remove(facet_col)

        if len(cols_not_taken) == 1:
            _color_name = None
            _x_name = str(cols_not_taken[0])
        else:
            raise ValueError(f"Input tensor has too many dimensions: {array.shape}. Available cols {cols_not_taken}")

        if color is not None:
            _color_name = color_name
            color = to_numpy(color)
            color = broadcast_up(color, array.shape)
            df[_color_name] = color.flatten()
    if x is not None:
        x = to_numpy(x)
        x = broadcast_up(x, array.shape)
        df[_x_name] = x.flatten()

    if custom_kwargs["hover"] is not None:
        # TODO: Add support for multi-hover
        hover_data = to_numpy(custom_kwargs["hover"])
        df[custom_kwargs["hover_name"]] = broadcast_up(hover_data, array.shape)
        hover_names = [custom_kwargs["hover_name"]]
    else:
        hover_names = []

    if plot_type == "line":
        plot_fn = px.line
    elif plot_type == "scatter":
        plot_fn = px.scatter
    else:
        raise ValueError

    fig = plot_fn(
        df,
        x=_x_name,
        y="value",
        color=_color_name,
        animation_frame=_animation_name,
        facet_col=_facet_col,
        hover_data=hover_names,
        labels={
            _x_name: xaxis,
            "value": yaxis,
            _color_name: color_name,
            _animation_name: animation_name,
            _facet_col: facet_name,
        },
        **plotly_kwargs,
    )
    if custom_kwargs['include_diag']:
        if facet_col is None:
            max_value = array.max()
            min_value = array.min()
            if x is not None:
                max_value = max(max_value, x.max())
                min_value = min(min_value, x.min())
            fig.add_shape(
                type="line",
                x0=min_value,
                y0=min_value,
                x1=max_value,
                y1=max_value,
                line={
                    "color": "gray",
                    "dash": "dash",
                },
                opacity=0.3
            )
        else:
            for col in range(array.shape[facet_col]):
                max_value = array.max()
                min_value = array.min()
                if x is not None:
                    max_value = max(max_value, x.max())
                    min_value = min(min_value, x.min())
                fig.add_shape(
                    type="line",
                    x0=min_value,
                    y0=min_value,
                    x1=max_value,
                    y1=max_value,
                    line={
                        "color": "gray",
                        "dash": "dash",
                    },
                    opacity=0.3,
                    col=col + 1,
                    row=1,
                )

    if _animation_name is not None:
        if custom_kwargs["animation_maxrange_x"] and x is not None:
            fig.layout.xaxis.range = [x.min(), x.max()]
        if custom_kwargs["animation_maxrange_y"]:
            fig.layout.yaxis.range = [array.min(), array.max()]

    update_fig(fig, custom_kwargs)

    if custom_kwargs["return_fig"]:
        return fig
    else:
        fig.show()


def scatter(x, y, filename, *args, **kwargs):
    return line_or_scatter(y, filename, x, *args, plot_type="scatter", **kwargs)


scatter = partial(scatter, color_continuous_scale="Portland")

line = partial(line_or_scatter, plot_type="line")