from typing import Union, List

# from loguru import logger
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from tqdm.auto import tqdm
# from utils import relpath_under
from scipy.stats import multivariate_normal
from plotly.subplots import make_subplots
from rose_colormap.plotly import rose_vivid
from rose_colormap import rose_vivid as rose_vivid_matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


from itertools import product, combinations

def _tile_3d_axes(fig, nrows, ncols,
                  left=0.06, right=0.96,
                  bottom=0.18, top=0.92):
    """
    This function is used to tile the 3D axes in the figure.
    Rearrange the 3D axes in the figure (in row-major order) into a tight grid of nrows × ncols, no row/column gaps.
    left/right/bottom/top are the boundaries of the entire panel area in figure coordinates.
    """
    panel_axes = [ax for ax in fig.axes if getattr(ax, "name", "") == "3d"]
    assert len(panel_axes) >= nrows * ncols, "3D axes fewer than expected"

    panel_axes = panel_axes[: nrows * ncols]

    cell_w = (right - left) / ncols
    cell_h = (top - bottom) / nrows

    # row = 0 is the top row
    for r in range(nrows):
        for c in range(ncols):
            ax = panel_axes[r * ncols + c]
            x0 = left + c * cell_w
            y0 = bottom + (nrows - 1 - r) * cell_h  # The bottom row has smaller y
            ax.set_position([x0, y0, cell_w, cell_h])



def add_time_axis(fig, times, label="time", t_max=None,
                          stub_left_frac=0.25, stub_right_frac=0.25):
    """
    This function is used to add a time axis to the figure.
    Args:
        fig: the figure to add the time axis
        times: the times to add the time axis
        label: the label of the time axis
        t_max: the maximum time of the time axis
        stub_left_frac: the fraction of the left stub
        stub_right_frac: the fraction of the right stub
    """
    times = np.asarray(times, dtype=float)
    K = len(times)
    assert K >= 1

    # Find all 3D axes
    panel_axes = [ax for ax in fig.axes if getattr(ax, "name", "") == "3d"]
    if not panel_axes:
        raise RuntimeError(
            "No 3D axes found in figure; call this after plot_lambst_panels."
        )

    # Calculate the x coordinates of the column centers of each 3D axis (figure coordinates)
    centers = []
    for ax in panel_axes:
        bb = ax.get_position()  # Bbox in figure coordinates
        xc = 0.5 * (bb.x0 + bb.x1)
        centers.append(xc)

    centers = sorted(set(centers))
    if len(centers) != K:
        # It should usually be equal; if not, truncate to the smaller side
        K_eff = min(len(centers), K)
        centers = centers[:K_eff]
        times = times[:K_eff]
        K = K_eff

    center_min = centers[0]
    center_max = centers[-1]
    span = center_max - center_min if K > 1 else 1.0

    # Determine the left and right range of the time axis on the figure
    if t_max is not None:
        stub_left = stub_left_frac * span
        stub_right = stub_right_frac * span
    else:
        stub_left = 0.0
        stub_right = 0.0

    left = center_min - stub_left
    right = center_max + stub_right

    height = 0.08
    bottom = 0.02

    # Create a new bottom axis: data-x directly uses figure-x, ensuring alignment
    ax_time = fig.add_axes([left, bottom, right - left, height])
    ax_time.set_axis_off()
    ax_time.set_xlim(left, right)
    ax_time.set_ylim(-1.0, 1.0)

    # Main axis line: 0 line
    ax_time.hlines(0.0, left, right, linewidth=3.0, color="black")

    # Left 0, right t_max
    if t_max is not None:
        # Left 0
        ax_time.vlines(left, 0.0, -0.18, linewidth=2.0, color="black")
        ax_time.text(
            left, -0.35, "0",
            ha="center", va="top", fontsize=18,
        )

    # Middle columns corresponding ticks & text (still aligned with the center of the 3D graph)
    for xc, t in zip(centers, times):
        ax_time.vlines(xc, 0.0, -0.18, linewidth=2.0, color="black")
        ax_time.text(
            xc, -0.35, f"{t:.0f}",
            ha="center", va="top", fontsize=18,
        )

    if t_max is not None:
        # Right t_max
        ax_time.vlines(right, 0.0, -0.18, linewidth=2.0, color="black")
        ax_time.text(
            right, -0.35, f"{t_max:.0f}",
            ha="center", va="top", fontsize=18,
        )

        # Arrow starts from the column closest to the last column, pointing to the right
        arrow_start = center_max + 0.10 * span
        arrow_end = right
    else:
        # If there is no t_max, draw a short arrow near the last column
        arrow_start = center_max - 0.15 * span
        arrow_end = center_max + 0.05 * span

    ax_time.annotate(
        "",
        xy=(arrow_end, 0.0),
        xytext=(arrow_start, 0.0),
        arrowprops=dict(arrowstyle="->", linewidth=3.0),
        clip_on=False,
    )

    # "time" label
    ax_time.text(
        arrow_end + 0.03 * span,
        -0.35,
        label,
        ha="left",
        va="top",
        fontsize=18,
    )

    return ax_time

def draw_cube_bbox(ax, xlim, ylim, zlim, color='k', lw=0.4):
    """
    This function is used to draw a cube bounding box on the given x/y/z range for the 3D axis ax.
    """
    xs = [xlim[0], xlim[1]]
    ys = [ylim[0], ylim[1]]
    zs = [zlim[0], zlim[1]]

    # Cartesian product of 8 corners
    corners = np.array(list(product(xs, ys, zs)))

    # Pairwise combinations, if only one coordinate is different -> one edge
    for s, e in combinations(corners, 2):
        if np.sum(s != e) == 1:  # Only one coordinate is different
            ax.plot3D(*zip(s, e), color=color, lw=lw)

def visualize_diff(outputs, targets, portion=1., fn=None):
    """
    Plot and compare the event trajectories in an index-aligned fashion
    assuming outputs is a time series with 3 features, (lat, lon, delta_time)
    or 1-step sliding windows of such a series with (lookahead) windows

    :param outputs: [batch, lookahead, 3] or [batch, 3]
    :param targets: [batch, lookahead, 3] or [batch, 3]
    :param portion: portion of outputs to be visualized, 0. ~ 1.
    :param fn: the saving filename
    :return: fig and axis
    """
    if len(targets.shape) == 2:
        outputs = np.expand_dims(outputs, 1)
        targets = np.expand_dims(targets, 1)

    outputs = outputs[:int(len(outputs) * portion)]
    targets = targets[:int(len(targets) * portion)]

    plt.figure(figsize=(14, 10), dpi=180)
    plt.subplot(2, 2, 1)

    n = outputs.shape[0]
    lookahead = outputs.shape[1]

    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 0], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 0], "-o", color="b", label="Actual")
    plt.ylabel('Latitude')
    plt.legend()

    plt.subplot(2, 2, 2)
    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 1], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 1], "-o", color="b", label="Actual")
    plt.ylabel('Longitude')
    plt.legend()

    plt.subplot(2, 2, 3)
    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 2], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 2], "-o", color="b", label="Actual")
    plt.ylabel('delta_t (hours)')
    plt.legend()

    if fn is not None:
        plt.savefig(fn)

    return plt.gcf(), plt.gca()


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration},
    }


def inverse_transform(x_range, y_range, t_range, scaler):
    """
    x_range, y_range, t_range: 1D array of any length
    """
    # Inverse transform the data
    temp = np.zeros((len(x_range), 3))
    temp[:, 0] = x_range
    x_range = scaler.inverse_transform(temp)[:, 0]

    temp = np.zeros((len(y_range), 3))
    temp[:, 1] = y_range
    y_range = scaler.inverse_transform(temp)[:, 1]

    temp = np.zeros((len(t_range), 3))
    temp[:, 2] = t_range
    t_range = scaler.inverse_transform(temp)[:, 2]

    return x_range, y_range, t_range


def plot_lambst_static(lambs, x_range, y_range, t_range, fps, scaler=None, cmin=None, cmax=None,
                       history=None, decay=0.3, base_size=300, cmap=rose_vivid_matplotlib, fn='result.mp4'):
    """
    The result could be saved as file with command `ani.save('file_name.mp4', writer='ffmpeg', fps=fps)`
                                        or command `ani.save('file_name.gif', writer='imagemagick', fps=fps)`

    :param lambs: list, len(lambs) = len(t_range), element: [len(x_range), len(y_range)]
    :param fps: # frame per sec
    :param fn: file_name
    """
    if type(t_range) is torch.Tensor:
        t_range = t_range.numpy()
    
    # Inverse transform the range to the actual scale
    if scaler is not None:
        x_range, y_range, t_range = inverse_transform(x_range, y_range, t_range, scaler)

    if cmin is None:
        cmin = 0
    if cmax == "outlier":
        cmax = np.max([np.max(lamb_st) for lamb_st in lambs])
    if cmax is None:
        cmax = np.max(lambs)
    # logger.debug(f'Inferred cmax: {cmax}')
    cmid = cmin + (cmax - cmin) * 0.9

    grid_x, grid_y = np.meshgrid(x_range, y_range)

    frn = len(t_range)  # frame number of the animation

    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d', xlabel='x', ylabel='y', zlabel='λ', zlim=(cmin, cmax),
                         title='Spatio-temporal Conditional Intensity')
    ax.set_box_aspect([1, 1, 1])
    ax.title.set_position([.5, .95])
    text = ax.text(min(x_range), min(y_range), cmax, "t={:.2f}".format(t_range[0]), fontsize=10)
    plot = [ax.plot_surface(grid_x, grid_y, lambs[0], rstride=1, cstride=1, cmap=cmap)]

    if history is not None:
        his_s, his_t = history
        zs = np.ones_like(lambs[0]) * cmid  # add a platform for locations
        plat = ax.plot_surface(grid_x, grid_y, zs, rstride=1, cstride=1, color='white', alpha=0.2)
        points = ax.scatter3D([], [], [], color='black')  # add locations
        plot.append(plat)
        plot.append(points)

    pbar = tqdm(total=frn + 2)

    def update_plot(frame_number):
        t = t_range[frame_number]
        plot[0].remove()
        plot[0] = ax.plot_surface(grid_x, grid_y, lambs[frame_number], rstride=1, cstride=1, cmap=cmap)
        text.set_text('t={:.2f}'.format(t))

        if history is not None:
            mask = np.logical_and(his_t <= t, his_t >= t_range[0])
            locs = his_s[mask]
            times = his_t[mask]
            sizes = np.exp((times - t) * decay) * base_size
            zs = np.ones_like(sizes) * cmid
            plot[2].remove()
            plot[2] = ax.scatter3D(locs[:, 0], locs[:, 1], zs, c='black', s=sizes, marker='x')

        pbar.update()

    ani = animation.FuncAnimation(fig, update_plot, frn, interval=1000 / fps)
    ani.save(fn, writer='ffmpeg', fps=fps)
    return ani


def plot_lambst_interactive(lambs: Union[List, np.array], x_range, y_range, t_range, cmin=None, cmax=None,
                            scaler=None, heatmap=False, colorscale=rose_vivid, show=True, cauto=False,
                            master_title='Spatio-temporal Conditional Intensity', subplot_titles=None, name=None):
    """
    :param lambs:   3D Array-like sampled intensities of shape (t_range, x_range, y_range)
                    or 4D Array-like of shape (N, ...) to compare plots side-by-side
    :param x_range: 1D Array-like, specifying sampling x's locations
    :param y_range: 1D Array-like, specifying sampling y's locations
    :param t_range: 1D Array_like, specifying sampling t's locations
    :param cmin: lower bound of lambs axis, 0 if unspecified
    :param cmax: upper bound of lambs axis, max(lambs) if unspecified
    :param scaler: scipy.MinMaxScaler, used for scaling the intensities
    :param heatmap: whether draw the intensities as a heatmap instead of 3D surface plot
    :param colorscale: Color scales used for surface
    :param show: whether to show the figure
    :param master_title: the one title above all
    :param subplot_titles: 1D Array of N str, title of each side-by-side comparison plot
    """
    assert type(colorscale) == list and type(colorscale[0][1]) == str, "Unrecognized colorscale"

    if scaler is not None:  # Inverse transform the range to the actual scale
        x_range, y_range, t_range = inverse_transform(x_range, y_range, t_range, scaler)

    n_subplot = 1
    if type(lambs) == list:  # Convert lists to numpy array
        lambs = np.array(lambs)

    # Shape checks
    if len(lambs.shape) == 4:
        n_subplot = len(lambs)
        lambs_shape = lambs.shape[1:]
        lambs = lambs.transpose([1, 0, 2, 3])  # Put time before N
        assert subplot_titles is None or len(subplot_titles) == n_subplot
    else:
        assert len(lambs.shape) == 3
        lambs_shape = lambs.shape
    assert lambs_shape == (len(t_range), len(x_range), len(y_range))

    if cmin is None:
        cmin = min(0, np.min(lambs))
    if cmax is None:
        cmax = np.max(lambs)
    frames = []

    for i, lamb_st in enumerate(lambs):
        if n_subplot != 1:
            data = []
            for j, lamb_st_i in enumerate(lamb_st):
                if heatmap:
                    data.append(go.Heatmap(z=lamb_st_i, x=x_range, y=y_range, zmin=cmin, zmax=cmax,
                                           colorscale=colorscale))
                else:
                    data.append(go.Surface(z=lamb_st_i, x=x_range, y=y_range, cmin=cmin, cmax=cmax,
                                           colorscale=colorscale))
            frames.append(go.Frame(data=data, name="{:.2f}".format(t_range[i])))
        else:
            if heatmap:
                data = go.Heatmap(z=lamb_st, x=x_range, y=y_range, zmin=cmin, zmax=cmax, 
                                  colorscale=colorscale)
            else:
                data = go.Surface(z=lamb_st, x=x_range, y=y_range, cmin=cmin, cmax=cmax, 
                                  colorscale=colorscale)
            frames.append(go.Frame(data=data, name="{:.2f}".format(t_range[i])))

    if n_subplot != 1:
        if heatmap:
            specs = [[{"type": "xy"}] * n_subplot]
        else:
            specs = [[{"type": "scene"}] * n_subplot]
        fig = make_subplots(rows=1, cols=n_subplot, horizontal_spacing=0.05,
                            specs=specs, subplot_titles=subplot_titles)
        fig.frames = frames
    else:
        fig = go.Figure(frames=frames)

    # Add data to be displayed before animation starts
    if n_subplot != 1:
        for j, lamb_st_i in enumerate(lambs[0]):
            if heatmap:
                fig.add_trace(go.Heatmap(z=lamb_st_i, x=x_range, y=y_range, zmin=cmin, zmax=cmax,
                                         colorscale=colorscale), row=1, col=j + 1)
            else:
                fig.add_trace(go.Surface(z=lamb_st_i, x=x_range, y=y_range, cmin=cmin, cmax=cmax,
                                         colorscale=colorscale), row=1, col=j + 1)
    else:
        if heatmap:
            fig.add_trace(go.Heatmap(z=lambs[0], x=x_range, y=y_range, zmin=cmin, zmax=cmax, 
                                     colorscale=colorscale))
        else:
            fig.add_trace(go.Surface(z=lambs[0], x=x_range, y=y_range, cmin=cmin, cmax=cmax, 
                                     colorscale=colorscale))

    # Slider
    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": f.name,
                    "method": "animate",
                }
                for f in fig.frames
            ],
        }
    ]

    fig.update_scenes(  # Control zaxis for all subplots
        aspectmode='cube',
        zaxis_title='λ',
        zaxis=dict(range=[cmin, cmax], autorange=False),
        # camera=dict(
        #     eye=dict(x=1, y=-1.73205, z=1.15470)
        # )
    )

    # Layout
    fig.update_layout(
        title=master_title,
        width=500 * n_subplot + 180,
        height=700,
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(1)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )

    # save the figure
    fig.write_html(f"./htmls/interactive{name}.html")
    print(f"Saving intensities to ./htmls/interactive{name}.html...")
    # if show:
    #     fig.show()
    return fig



def plot_lambst_panels(
    lambs,
    T_max,
    x_range,
    y_range,
    t_range,
    t_indices=None,          # Draw which time index; if not specified, take 4 equally spaced ones
    model_names=None,        # Each row name, e.g. ['GT', 'STH', 'N-STH']
    cmap=None,               # Default use viridis; you can pass rose_vivid_matplotlib
    cmin=None,
    cmax=None,
    figsize=None,            # Default auto-scale by number of columns/rows
    dpi=600,                 # Default resolution set to 600
    elev=35,
    azim=-60,
    savepath="sth_panels.pdf",   # Default save as pdf
    n_xy_ticks=5, n_z_ticks=5,
    # ---- History event cross-related parameters ----
    history=None,            # (his_s, his_t), or None; his_s shape (N_evt, 2), his_t shape (N_evt,)
    history_rows=(0,),       # Which rows to draw crosses: default only on top row
    history_decay=0.3,       # The farther the time, the smaller the cross: exp((time - t)*decay)
    history_base_size=40.0,  # Base size of crosses
    history_z_ratio=0.9,     # Z plane of crosses: cmin + ratio*(cmax-cmin)
):
    """
    This function is used to plot the spatio-temporal conditional intensities in a panel.
    Args:
        lambs: shape = (T, X, Y) or (N, T, X, Y)
        T_max: the maximum time of the time axis
        x_range: the range of the x axis
        y_range: the range of the y axis
        t_range: the range of the time axis
        t_indices: the indices of the time axis to draw
        model_names: the names of the models
        cmap: the colormap to use
        cmin: the minimum value of the lambs axis
    """

    # ===== 字体与字号统一设置 =====
    base_fontsize = 11        # Global base font size
    axis_label_fs = 14        # x,y axis label font size
    tick_label_fs = 14        # Coordinate axis tick font size
    model_name_fs = 18        # Left GT/DSM/WSM label font size for each row
    lambda_label_fs = 24     # Global λ label font size (one size larger than the axis label)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "mathtext.default": "it",
        "font.size": base_fontsize,
    })

    # if cmap is None:
    #     cmap = cm.viridis  # If you want rose_vivid_matplotlib, pass it when calling
    if cmap is None: 
        cmap = rose_vivid_matplotlib

    lambs = np.asarray(lambs)
    if lambs.ndim == 3:
        lambs = lambs[None, ...]   # -> (1, T, X, Y)
    assert lambs.ndim == 4
    N, T, X, Y = lambs.shape

    x_range = np.asarray(x_range)
    y_range = np.asarray(y_range)
    t_range = np.asarray(t_range)
    assert X == len(x_range) and Y == len(y_range) and T == len(t_range)

    # Which frames to draw
    if t_indices is None:
        ncols = min(4, T)
        t_indices = np.linspace(0, T - 1, ncols, dtype=int)
    else:
        t_indices = np.asarray(t_indices, dtype=int)
        ncols = len(t_indices)

    nrows = N
    if figsize is None:
        figsize = (3.3 * ncols, 3.8 * nrows)

    if cmin is None:
        cmin = float(lambs.min())
    if cmax is None:
        cmax = float(lambs.max())

    # Shared λ axis ticks
    zticks = np.linspace(cmin, cmax, n_z_ticks)

    # History preprocessing
    if history is not None:
        his_s, his_t = history
        his_s = np.asarray(his_s)
        his_t = np.asarray(his_t)
        assert his_s.shape[1] == 2
        assert his_s.shape[0] == his_t.shape[0]
        cmid = cmin + (cmax - cmin) * float(history_z_ratio)
    else:
        his_s = his_t = cmid = None

    Xg, Yg = np.meshgrid(x_range, y_range, indexing='ij')
    xticks = np.linspace(x_range.min(), x_range.max(), n_xy_ticks)
    yticks = np.linspace(y_range.min(), y_range.max(), n_xy_ticks)

    fig = plt.figure(figsize=figsize)

    for row in range(nrows):
        for col, ti in enumerate(t_indices):
            ax = fig.add_subplot(nrows, ncols, row * ncols + col + 1,
                                 projection='3d')

            Z = lambs[row, ti]

            ax.plot_surface(
                Xg, Yg, Z,
                cmap=cmap,
                vmin=cmin, vmax=cmax,
                linewidth=0,
                antialiased=True,
                shade=True,
                rcount=80, ccount=80,
            )

            ax.view_init(elev=elev, azim=azim)
            ax.dist = 8.  # Smaller to make the cube fill the axes

            # ===== z axis: only keep ticks in the left column, other columns don't write =====
            ax.set_zlim(cmin, cmax)
            ax.set_zticks(zticks)

            if col == 0:
                # Left column: z axis on the left, keep tick numbers, don't write λ
                ax.zaxis._axinfo['juggled'] = (1, 2, 0)
                ax.set_zlabel("")
                ax.set_zticklabels([f"{v:.1f}" for v in zticks],
                                   fontsize=tick_label_fs)
            else:
                # Other columns: z axis on the right, don't write tick labels
                ax.zaxis._axinfo['juggled'] = (0, 2, 1)
                ax.set_zlabel("")
                ax.set_zticklabels([])

            # ===== y axis: only keep ticks in the bottom row, other rows don't write =====
            ax.set_yticks(yticks)
            if row == nrows - 1:
                ax.set_ylabel("y", labelpad=8, fontsize=axis_label_fs)
                ax.set_yticklabels([f"{v:g}" for v in yticks], fontsize=tick_label_fs)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            # ===== x axis: also only keep ticks in the bottom row, other rows don't write =====
            ax.set_xticks(xticks)
            if row == nrows - 1:
                ax.set_xlabel("x", labelpad=8, fontsize=axis_label_fs)
                ax.set_xticklabels([f"{v:g}" for v in xticks], fontsize=tick_label_fs)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            # Uniform tick font size
            ax.tick_params(axis='x', labelsize=tick_label_fs)
            ax.tick_params(axis='y', labelsize=tick_label_fs)
            ax.tick_params(axis='z', labelsize=tick_label_fs)

            # Each row left model name (GT / MLE / DSM / WSM)
            if col == 0 and model_names is not None:
                ax.text2D(
                    -0.30, 0.5, model_names[row],
                    transform=ax.transAxes,
                    rotation=90,
                    va='center',
                    ha='center',
                    fontsize=model_name_fs,
                )

            # ===== History event crosses =====
            if (history is not None) and (row in history_rows):
                t_cur = t_range[ti]
                mask = his_t <= t_cur    # Points before the current time
                if np.any(mask):
                    locs = his_s[mask]
                    times = his_t[mask]
                    sizes = np.exp((times - t_cur) * history_decay) * history_base_size
                    z_hist = np.full_like(times, cmid, dtype=float)
                    ax.scatter(
                        locs[:, 0],
                        locs[:, 1],
                        z_hist,
                        c='k',
                        s=sizes,
                        marker='x',
                        linewidths=0.8,
                    )

            # Grid + cube aspect ratio (to make multiple rows look connected)
            ax.grid(True)
            ax.set_xlim(x_range.min(), x_range.max())
            ax.set_ylim(y_range.min(), y_range.max())
            ax.set_box_aspect((1, 1, 0.85))

            # ---------- Make the pane transparent, only keep the grid ----------
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.pane.fill = False
                axis.pane.set_alpha(0.0)

    # ===== Stack all 3D subplots into a "tower" =====
    _tile_3d_axes(
        fig, nrows, ncols,
        left=0.06, right=0.96,
        bottom=0.15, top=0.95
    )

    # ===== Global λ label: placed on the left, vertically centered =====
    # Find the leftmost 3D axis, use its x0 as reference
    three_d_axes = [ax for ax in fig.axes if getattr(ax, "name", "") == "3d"]
    if three_d_axes:
        leftmost_ax = min(three_d_axes, key=lambda a: a.get_position().x0)
        pos = leftmost_ax.get_position()
        # Move slightly to the left; ensure it doesn't go out of the figure (>= 0.01)
        x_text = max(pos.x0 - 0.06, 0.01)
    else:
        # Fallback: use a position slightly to the left
        x_text = 0.03

    # y=0.5: vertically centered on the entire figure (even rows are in the middle of two rows)
    fig.text(
        x_text, 0.5,
        r"$\lambda$",
        ha="center", va="center",
        fontsize=lambda_label_fs,
        rotation=0,          # Ensure λ is "upright", not rotated with the axis
    )

    # ❌ Don't draw colorbar

    # time axis
    times_for_axis = np.array(t_range)
    add_time_axis(fig, times_for_axis, label="time", t_max=T_max)

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {savepath}")

    return fig


