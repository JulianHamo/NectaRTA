## Import

import h5py
import os
import numpy as np
import time
import pandas as pd
from pathlib import Path

from bokeh.models import (
ColumnDataSource, HoverTool,
Slider, Div, Select, Switch,
TabPanel, Tabs, Ellipse, Plot,
Range1d, AnnularWedge, Legend,
LegendItem)
from bokeh.transform import linear_cmap
from bokeh.palettes import Inferno
from bokeh.layouts import column, row, gridplot
from bokeh.plotting import curdoc, figure, show

from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization.bokeh import CameraDisplay, BokehPlot

## Variables

geom = CameraGeometry.from_name("NectarCam-003")
geom = geom.transform_to(EngineeringCameraFrame())

default_n_bins = 20
default_n_runs = 1
real_time_tag = "Real time"

DISPLAY_REGISTRY = []
WIDGETS = {}

## Function

## Header

def get_latest_file(ressource_path):
    filepath = max(Path(ressource_path).glob("*.h5"), key=lambda f: f.stat().st_mtime)
    file = h5py.File(filepath, "r")
    return file

def list_runs(ressource_path):
    return [item[:-3] for item in os.listdir(ressource_path) if item.endswith(".h5")]

def make_select_run(list_file):
    return Select(
        title="Run selected:",
        value="Real time",
        options=list(list_file) + ["Real time"]
    )

def make_status_div(file):
    file_div = Div(text=f"Loaded file: {file.filename}")
    time_div = Div(text=f"Last update: {time.strftime('%H:%M:%S')}")
    return column(file_div, time_div)

def make_header_menu(ressource_path, file=None):
    if file is None:
        file = get_latest_file(ressource_path)
    list_file = list_runs(ressource_path)
    run_choice_slidedown = make_select_run(list_file)

    #def _on_select_change(attr, old, new):
    #    new_filename = run_choice_slidedown.value
    #    if new_filename == real_time_tag:
    #        filepath = max(Path(ressource_path).glob("*.h5"), key=lambda f: f.stat().st_mtime)
    #    else:
    #        filepath = os.path.join(ressource_path, new_filename + ".h5")
    #    file = h5py.File(filepath, "r")
    #    return file

    #run_choice_slidedown.on_change("value", _on_select_change)
    #file = _on_select_change(None, None, None)
        
    status_div = make_status_div(file)
    return run_choice_slidedown, status_div

## Summary card

def make_summary_card(
    file,
    run=None,
    parentkeys={
        "parameter_parentkey":"dl1/event/telescope/parameters/tel_001",
        "layout_parentkey":"configuration/instrument/subarray/layout",
        "trigger_parentkey":"dl1/event/subarray/trigger"
    },
    childkeys={
        "tel_id":"tel_id",
        "event_id":"event_id",
        "name":"name",
        "camera_type":"camera_type",
        "time":"time",
        "timestamp_qns":"timestamp_qns",
        "alt_tel":"alt_tel",
        "az_tel":"az_tel",
        "obs_id":"obs_id",
        "event_type":"event_type",
        "event_quality":"event_quality",
        "is_good_event":"is_good_event",
    }
):
    if run is None:
        run = -1

    tel_id = file[parentkeys["parameter_parentkey"]][childkeys["tel_id"]][run]
    event_id = file[parentkeys["parameter_parentkey"]][childkeys["event_id"]][run]
    layout_id = np.where(file[parentkeys["layout_parentkey"]][childkeys["tel_id"]] == tel_id)[0][0]
    trigger_id = np.where(file[parentkeys["trigger_parentkey"]][childkeys["event_id"]] == event_id)[0][0]
    name = file[parentkeys["layout_parentkey"]][childkeys["name"]][layout_id].decode("utf-8")
    camera_type = file[parentkeys["layout_parentkey"]][childkeys["camera_type"]][layout_id].decode("utf-8")
    time = file[parentkeys["trigger_parentkey"]][childkeys["time"]][trigger_id]
    timestamp_qns = file[parentkeys["trigger_parentkey"]][childkeys["timestamp_qns"]][trigger_id]
    alt_tel = file[parentkeys["parameter_parentkey"]][childkeys["alt_tel"]][run]
    az_tel = file[parentkeys["parameter_parentkey"]][childkeys["az_tel"]][run]
    obs_id = file[parentkeys["parameter_parentkey"]][childkeys["obs_id"]][run]
    event_type = file[parentkeys["parameter_parentkey"]][childkeys["event_type"]][run]
    event_quality = file[parentkeys["parameter_parentkey"]][childkeys["event_quality"]][run]
    event_goodness = file[parentkeys["parameter_parentkey"]][childkeys["is_good_event"]][run]
    
    title = Div(
        text="<strong>Summary card:</strong>",
    )
    card = Div(
        text=f"""
            <div>
              Telescope: {name} - id: {tel_id}<br>
              Camera: {camera_type}<br>
              Position:<br>
              <dl>
              <dd>altitude: {alt_tel}</dd>
              <dd>azimut: {az_tel}</dd>
              </dl>
              Observation: {obs_id}<br>
              Event:<br>
              <dl>
              <dd>event type: {event_type}</dd>
              <dd>id: {event_id}</dd>
              <dd>event quality: {event_quality}</dd>
              <dd>event goodness: {event_goodness}</dd>
              </dl>
              Observation time: {time} [units]<br>
              Timestamp: {timestamp_qns / 1e9} [s]<br>
            </div>
        """,
        styles={
            "border": "2px solid #e2e8f0",
            "padding": "12px",
            "border-radius": "8px",
            "width": "300px",
        },
        width=300, height=300
    )
    return column(title, card)

## Camera display

def make_camera_display_params(show_hillas=False, label=None):
    if label is None:
        label = "Show Hillas ellipse:"
    return Switch(active=show_hillas, label=label)

def make_camera_display(
    file,
    childkey,
    image_parentkey="dl1/event/telescope/images/tel_001",
    parameterkeys={
        "parentkey":"dl1/event/telescope/parameters/tel_001",
        "hillas_x_key":"hillas_x",
        "hillas_y_key":"hillas_y",
        "hillas_length_key":"hillas_length",
        "hillas_width_key":"hillas_width",
        "hillas_phi_key":"hillas_phi",
    },
    run=None,
    title=None,
    show_hillas=False,
    label_colorbar=None
):
    if run is None:
        run = -1
    if title is None:
        title = childkey
    image = file[image_parentkey][childkey]
    image = np.nan_to_num(image[run], nan=0.0)
    display = CameraDisplay(geometry=geom)
    try:
        display.image = image
    except ValueError:
        print("ValueError")
        image = np.zeros(shape=display.image.shape)
        display.image = image
    except KeyError:
        print("KeyError")
        image = np.zeros(shape=constants.N_PIXELS)
        display.image = image
    ellipse = Ellipse(
        x=file[parameterkeys["parentkey"]][parameterkeys["hillas_x_key"]][run],
        y=file[parameterkeys["parentkey"]][parameterkeys["hillas_y_key"]][run],
        width=file[parameterkeys["parentkey"]][parameterkeys["hillas_length_key"]][run],
        height=file[parameterkeys["parentkey"]][parameterkeys["hillas_width_key"]][run],
        angle=file[parameterkeys["parentkey"]][parameterkeys["hillas_phi_key"]][run],
        fill_color=None,
        line_color="#40E0D0",
        line_width=2,
        line_alpha=1
    )
    glyph = display.figure.add_glyph(ellipse)
    hovertool = [t for t in display.figure.tools if isinstance(t, HoverTool)][0]
    hovertool.renderers = [display.figure.renderers[0]]
    
    if not show_hillas:
        glyph.visible = False
    display._annotations.append(glyph)
    display.update()
    display.add_colorbar()
    display._color_bar.title = label_colorbar
    display.figure.title = title
    
    display._meta = {
        "type": "camera",
        "image_parentkey": image_parentkey,
        "childkey": childkey,
        "parameterkeys": parameterkeys,
        "factory": "make_camera_display"
    }
    DISPLAY_REGISTRY.append(display)
    
    return display
    

def make_tab_camera_displays(
    file,
    childkeys,
    image_parentkeys=None,
    parameterkeys=None,
    run=None,
    titles=None,
    show_hillas=False,
    label_hillas=None,
    labels_colorbar=None
):
    displays = []
    if titles is None:
        titles = childkeys
    if labels_colorbar is None:
        labels_colorbar = [None] * len(childkeys)
        
    for index in range(len(childkeys)):
        args = {
            "file": file,
            "childkey": childkeys[index],
            "run": run,
            "title": titles[index],
            "show_hillas": show_hillas,
            "label_colorbar": labels_colorbar[index]
            
        }
        for content, key in zip(
            [image_parentkeys, parameterkeys],
            ["image_parentkey", "parameterkeys"]
        ):
            if content is not None:
                try:
                    args["key"] = content[index]
                except:
                    pass
        displays.append(make_camera_display(**args).figure)

    hillas_switch = make_camera_display_params(
        show_hillas=show_hillas, label=label_hillas
    )
    WIDGETS["hillas_switch"] = hillas_switch
    
    def hillas_callback(attr, old, new):
        for display in displays:
            for r in display.renderers:
                if isinstance(r.glyph, Ellipse):
                    r.visible = not r.visible
            display.update()
    hillas_switch.on_change("active", hillas_callback)
    display_gridplot = gridplot(displays, ncols=2)
    
    display_layout = column(hillas_switch, display_gridplot)
    tab_camera_displays = TabPanel(child=display_layout, title="Camera displays")
    return tab_camera_displays

## Timelines

def make_2d_timeline(
    file,
    childkey,
    parentkey,
    ylabel=None,
    labels=None,
    funcs=None
):
    if funcs is None:
        funcs = [np.mean]
    if labels is None:
        labels = ["Mean"]
    if len(labels) != len(funcs):
        funcs = [np.mean]
        labels = ["Mean"]
    data = np.asarray(file[parentkey][childkey])
    if data.ndim != 2:
        data = np.zeros((data.shape[0],1))
    if ylabel is None:
        ylabel = childkey
    data_size = data.shape[0]
    
    display = BokehPlot(
        #title=title,
        tools=("xpan", "box_zoom", "wheel_zoom", "save", "reset"),
        active_drag="xpan",
        x_range=(0, data_size),
        toolbar_location="above"
    )
    fig = display.figure
    fig.tools = [t for t in fig.tools if not isinstance(t, HoverTool)]
    fig.xaxis.axis_label = "Time [unit time]"
    fig.yaxis.axis_label = ylabel.capitalize()

    computed_data = ColumnDataSource(
        data={
            "time": np.arange(data_size),     # Change for meaningful time
        }
        | {
            labels[index]: funcs[index](data, axis=-1)
            for index in range(len(labels))
        }
    )

    colors = Inferno[len(funcs)+2][1:-1]
    for index in range(len(funcs)):
        r = fig.line(
            source=computed_data,
            x="time",
            y=labels[index],
            line_width=2,
            name=labels[index],
            color=colors[index],
            alpha=1,
            muted_alpha=.2,
            legend_label=labels[index].capitalize()
        )
        hover = HoverTool(
            tooltips=[(labels[index], "@{}".format(labels[index]))],
            renderers=[r]
        )
        fig.add_tools(hover)

    fig.legend.location = "bottom_left"
    fig.legend.click_policy = "mute"
    fig.hover.mode = "vline"
    display.update()

    display._meta = {
        "type": "timeline_2d", 
        "parentkey": parentkey,
        "childkey": childkey,
        "funcs": funcs,       
        "labels": labels,
        "factory": "make_2d_timeline"
    }
    DISPLAY_REGISTRY.append(display)

    return display

def make_2d_timelines(
    file,
    childkeys,
    parentkeys,
    ylabels=[None],
    labels=[None],
    funcs=[[np.mean]],
    suptitle=None
):
    if isinstance(parentkeys, str):
        parentkeys = [parentkeys for dummy in range(len(childkeys))]
    if len(childkeys) != len(parentkeys):
        parentkeys = [parentkeys[0] for dummy in range(len(childkeys))]
    if len(ylabels) != len(childkeys):
        ylabels = childkeys
    if not isinstance(funcs, list):
        funcs = [[funcs] for dummy in range(len(childkeys))]
    elif not isinstance(funcs[0], list):
        funcs = [funcs for dummy in range(len(childkeys))]
    if not isinstance(labels, list):
        labels = [[labels] for dummy in range(len(childkeys))]
    elif not isinstance(labels[0], list):
        labels = [labels for dummy in range(len(childkeys))]
            
    displays = []
    for index in range(len(childkeys)):
        displays.append(
            make_2d_timeline(
                file,
                childkeys[index],
                parentkeys[index],
                ylabel=ylabels[index],
                labels=labels[index],
                funcs=funcs[index]
            ).figure
        )

    if suptitle is None:
        suptitle = "<strong>Subsection</strong>"
    else:
        suptitle = "<strong>" + suptitle + "</strong>"
    name = Div(text=suptitle)
    
    return column(name, gridplot(displays, ncols=2))

def make_1d_timeline(
    file,
    childkey,
    parentkey,
    ylabel=None,
    label=None,
    step=False
):
    if label is None:
        label = childkey
    data = np.asarray(file[parentkey][childkey])
    if data.ndim != 1:
        data = np.zeros((data.shape[0]))
    if ylabel is None:
        ylabel = childkey
    data_size = data.shape[0]
    display = BokehPlot(
        #title=title,
        tools=("xpan", "box_zoom", "wheel_zoom", "save", "reset"),
        active_drag="xpan",
        x_range=(0, data_size),
        toolbar_location="above"
    )
    fig = display.figure
    fig.tools = [t for t in fig.tools if not isinstance(t, HoverTool)]
    fig.xaxis.axis_label = "Time [unit time]"
    fig.yaxis.axis_label = ylabel.capitalize()

    column_data = ColumnDataSource(
        data={
            "time": np.arange(data_size),
            "y": data
        }
    )

    color = Inferno[3][1]
    if step:
        r = fig.step(
            source=column_data,
            x="time",
            y="y",
            line_width=2,
            name=label,
            color=color,
            alpha=1,
            muted_alpha=.2,
            legend_label=label.capitalize()
        )
    else:
        r = fig.line(
            source=column_data,
            x="time",
            y="y",
            line_width=2,
            name=label,
            color=color,
            alpha=1,
            muted_alpha=.2,
            legend_label=label.capitalize()
        )
    hover = HoverTool(
        tooltips=[(label, "@y")],
        renderers=[r]
    )
    fig.add_tools(hover)

    fig.legend.location = "bottom_left"
    fig.legend.click_policy = "mute"
    fig.hover.mode = "vline"
    display.update()

    display._meta = {
        "type": "timeline_1d",  
        "parentkey": parentkey,
        "childkey": childkey,
        "labels": label,
        "factory": "make_1d_timeline"
    }
    DISPLAY_REGISTRY.append(display)

    return display

def make_1d_timelines(
    file,
    childkeys,
    parentkeys,
    ylabels=[None],
    labels=[None],
    suptitle=None,
    step=False
):
    if isinstance(parentkeys, str):
        parentkeys = [parentkeys for dummy in range(len(childkeys))]
    if len(childkeys) != len(parentkeys):
        parentkeys = [parentkeys[0] for dummy in range(len(childkeys))]
    if len(ylabels) != len(childkeys):
        ylabels = childkeys
    if len(labels) != len(childkeys):
        labels = childkeys
        
    displays = []
    for index in range(len(childkeys)):
        try:
            displays.append(
                make_1d_timeline(
                    file,
                    childkeys[index],
                    parentkeys[index],
                    ylabel=ylabels[index],
                    label=labels[index],
                    step=step
                ).figure
            )
        except:
            continue
    if suptitle is None:
        suptitle = "<strong>Subsection</strong>"
    else:
        suptitle = "<strong>" + suptitle + "</strong>"
    name = Div(text=suptitle)
            
    return column(name, gridplot(displays, ncols=2))

def make_tab_timelines(
    file,
    childkeys_2d,
    parentkeys_2d,
    childkeys_1d,
    parentkeys_1d,
    childkeys_step,
    parentkeys_step,
    ylabels_2d=[None],
    labels_2d=[None],
    funcs=[[np.mean]],
    suptitle_2d=None,
    ylabels_1d=[None],
    labels_1d=[None],
    suptitle_1d=None,
    ylabels_step=[None],
    labels_step=[None],
    suptitle_step=None,
):
    timeline_2d_layout = make_2d_timelines(
        file,
        childkeys_2d,
        parentkeys_2d,
        ylabels=ylabels_2d,
        labels=labels_2d,
        funcs=funcs,
        suptitle=suptitle_2d
    )
    timeline_1d_layout = make_1d_timelines(
        file,
        childkeys_1d,
        parentkeys_1d,
        ylabels=ylabels_1d,
        labels=labels_1d,
        suptitle=suptitle_1d,
        step=False
    )
    timeline_step_layout = make_1d_timelines(
        file,
        childkeys_step,
        parentkeys_step,
        ylabels=ylabels_step,
        labels=labels_step,
        suptitle=suptitle_step,
        step=True
    )
    timeline_layout = column(timeline_2d_layout, timeline_1d_layout, timeline_step_layout)
    return TabPanel(child=timeline_layout, title="Timelines")

# Histograms

def _recompute_display_hist(figure, data, label, n_runs, n_bins):
    """Helper: recompute histogram for one display and update its source (same as before)."""

    arr = np.asarray(data)

    # normalize shapes
    if arr.ndim != 2:
        arr = np.zeros((arr.shape[0], 1))

    n_runs = max(1, int(n_runs))
    n_bins  = max(1, int(n_bins))

    sample = arr[-n_runs:].ravel()
    hist, edges = np.histogram(sample, bins=n_bins)
    hist //= n_runs
    counts = hist.astype(int)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # update the source in-place
    try:
        # adjust vbar width (if present) and x_range
        try:
            width = 0.9 * (centers[1] - centers[0]) if len(centers) > 1 else 1.0
            for r in figure.renderers:
                # vbar glyphs usually expose a 'width' attribute
                if hasattr(r.glyph, "width"):
                    r.data_source.data = {label: hist, "edges": centers}
                    r.glyph.width = width
            figure.x_range.start = centers.min() - width / 2 if len(centers) else 0
            figure.x_range.end = centers.max() + width / 2 if len(centers) else 1
        except Exception:
            pass
    except Exception as e:
        print(f"_recompute_display_hist: failed to update source: {e}")
    figure.update()

def make_averaged_histogram(
    file,
    childkey,
    parentkey,
    n_runs=1,
    n_bins=default_n_bins,
    title=None,
    label=None
):
    if title is None:
        title = childkey
    if label is None:
        label = childkey
    data = np.asarray(file[parentkey][childkey])
    if data.ndim != 2:
        data = np.zeros((data.shape[0],1))
    
    data_to_average = data[-n_runs:]
    hist, edges = np.histogram(data_to_average, n_bins)
    hist //= n_runs
    current_data = ColumnDataSource(
        data={
            label: hist,
            "edges": (edges[:-1] + edges[1:]) / 2,
        }
    )

    display = BokehPlot(
        title=title,
        tools=("xpan", "box_zoom", "wheel_zoom", "save", "reset"),
        #tooltips=[("Count", "@{}".format(label))],
        active_drag="xpan",
        x_range=(current_data.data["edges"].min(), current_data.data["edges"].max()),
        toolbar_location="above"
    )
    
    fig = display.figure
    fig.tools = [t for t in fig.tools if not isinstance(t, HoverTool)]
    fig.y_range.start = 0
    fig.xaxis.axis_label = label.capitalize()

    r = fig.vbar(
        source=current_data,
        x="edges",
        width=0.9 * (edges[1] - edges[0]),
        top=label,
        color=linear_cmap(
            label,
            "Inferno256",
            -.1 * np.max(hist),
            1.1 * np.max(hist)
        )
    )

    hover = HoverTool(
        tooltips=[("Count", "@{}".format(label))],
        renderers=[r]
    )
    fig.add_tools(hover)

    display._meta = {
        "type": "hist_avg",
        "parentkey": parentkey,
        "childkey": childkey,
        "label": label,
        "factory": "make_averaged_histogram"
    }
    DISPLAY_REGISTRY.append(display)
    
    return display

def make_section_averaged_histogram_runs_only(
    file,
    childkeys,
    parentkeys,
    n_runs=1,
    n_bins=50,
    titles=None,
    labels=None,
):
    """
    Create multiple averaged-histogram displays and a single slider that controls
    the number of runs to average. When the slider moves, all histograms are recomputed.

    - file: opened h5py-like file object
    - childkeys: list of dataset names under each parent
    - parentkeys: either a single parent key or list of parent keys
    - n_runs: initial number of runs to average
    - n_bins: number of bins to use for the histograms (kept fixed here)
    - titles, labels: optional lists (will be broadcast if single provided)
    """
    # normalize lists
    N = len(childkeys)
    if not isinstance(parentkeys, list):
        parentkeys = [parentkeys] * N
    if titles is None:
        titles = childkeys
    if labels is None:
        labels = childkeys
    if not isinstance(titles, list):
        titles = [titles] * N
    if not isinstance(labels, list):
        labels = [labels] * N

    # build displays
    displays = []
    for i in range(N):
        disp = make_averaged_histogram(
            file,
            childkey = childkeys[i],
            parentkey = parentkeys[i],
            n_runs = n_runs,
            n_bins = n_bins,
            title = titles[i],
            label = labels[i]
        )
        displays.append(disp)

    # layout: sliders above grid of figures
    display_gridplot = gridplot([d.figure for d in displays], ncols=2)
    return display_gridplot

def make_histogram(
    file,
    childkey,
    parentkey,
    n_bins=default_n_bins,
    title=None,
    label=None
):
    if title is None:
        title = childkey
    if label is None:
        label = childkey
    data = np.asarray(file[parentkey][childkey])
    if data.ndim != 1:
        data = np.zeros((data.shape[0]))
    
    hist, edges = np.histogram(data, n_bins)
    current_data = ColumnDataSource(
        data={
            label: hist,
            "edges": (edges[:-1] + edges[1:]) / 2,
        }
    )

    display = BokehPlot(
        title=title,
        tools=("xpan", "box_zoom", "wheel_zoom", "save", "reset"),
        #tooltips=[("Count", "@{}".format(label))],
        active_drag="xpan",
        x_range=(current_data.data["edges"].min(), current_data.data["edges"].max()),
        toolbar_location="above"
    )
    
    fig = display.figure
    fig.tools = [t for t in fig.tools if not isinstance(t, HoverTool)]
    fig.y_range.start = 0
    fig.xaxis.axis_label = label.capitalize()

    r = fig.vbar(
        source=current_data,
        x="edges",
        width=0.9 * (edges[1] - edges[0]),
        top=label,
        color=linear_cmap(
            label,
            "Inferno256",
            -.1 * np.max(hist),
            1.1 * np.max(hist)
        )
    )

    hover = HoverTool(
        tooltips=[("Count", "@{}".format(label))],
        renderers=[r]
    )
    fig.add_tools(hover)

    display._meta = {
        "type": "hist_1d",
        "parentkey": parentkey,
        "childkey": childkey,
        "label": label,
        "factory": "make_histogram"
    }
    DISPLAY_REGISTRY.append(display)
    
    return display

def make_histograms(
    file,
    childkeys,
    parentkeys,
    n_bins=50,
    titles=None,
    labels=None,
    suptitle=None
):
    N = len(childkeys)
    if not isinstance(parentkeys, list):
        parentkeys = [parentkeys] * N
    if titles is None:
        titles = childkeys
    if labels is None:
        labels = childkeys
    if not isinstance(titles, list):
        titles = [titles] * N
    if not isinstance(labels, list):
        labels = [labels] * N

    displays = []
    for i in range(N):
        disp = make_histogram(
            file,
            childkey = childkeys[i],
            parentkey = parentkeys[i],
            n_bins = n_bins,
            title = titles[i],
            label = labels[i]
        )
        displays.append(disp)

    if suptitle is None:
        suptitle = "<strong>Subsection</strong>"
    else:
        suptitle = "<strong>" + suptitle + "</strong>"
    name = Div(text=suptitle)

    # layout: sliders above grid of figures
    display_gridplot = gridplot([d.figure for d in displays], ncols=2)
    display_layout = column(name, display_gridplot)
    return display_layout

def _recompute_display_hist_for_1d(figure, data, label, n_bins):
    """Helper: recompute histogram for one display and update its source (same as before)."""

    arr = np.asarray(data)

    # normalize shapes
    if arr.ndim != 1:
        arr = np.zeros(arr.shape[0])

    n_bins  = max(1, int(n_bins))

    hist, edges = np.histogram(arr, bins=n_bins)
    counts = hist.astype(int)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # update the source in-place
    try:
        # adjust vbar width (if present) and x_range
        try:
            width = 0.9 * (centers[1] - centers[0]) if len(centers) > 1 else 1.0
            for r in figure.renderers:
                # vbar glyphs usually expose a 'width' attribute
                if hasattr(r.glyph, "width"):
                    r.data_source.data = {label: hist, "edges": centers}
                    r.glyph.width = width
            figure.x_range.start = centers.min() - width / 2 if len(centers) else 0
            figure.x_range.end = centers.max() + width / 2 if len(centers) else 1
        except Exception:
            pass
    except Exception as e:
        print(f"_recompute_display_hist: failed to update source: {e}")
    figure.update()

def make_histogram_sections(
    file,
    childkeys_avg,
    parentkeys_avg,
    childkeys_1d,
    parentkeys_1d,
    n_runs=1,
    n_bins=50,
    titles_avg=None,
    labels_avg=None,
    suptitle_avg=None,
    titles_1d=None,
    labels_1d=None,
    suptitle_1d=None
):
    histogram_avg_layout = make_section_averaged_histogram_runs_only(
        file,
        childkeys_avg,
        parentkeys_avg,
        n_runs=n_runs,
        n_bins=n_bins,
        titles=titles_avg,
        labels=labels_avg,
    )
    histogram_1d_layout = make_histograms(
        file,
        childkeys_1d,
        parentkeys_1d,
        n_bins=n_bins,
        titles=titles_1d,
        labels=labels_1d,
        suptitle=suptitle_1d
    )

    N = len(childkeys_avg)
    if not isinstance(parentkeys_avg, list):
        parentkeys_avg = [parentkeys_avg] * N
    N = len(childkeys_1d)
    if not isinstance(parentkeys_1d, list):
        parentkeys_1d = [parentkeys_1d] * N
    if titles_avg is None:
        titles_avg = childkeys_avg
    if labels_avg is None:
        labels_avg = childkeys_avg
    if not isinstance(titles_avg, list):
        titles_avg = [titles_avg] * N
    if not isinstance(labels_avg, list):
        labels_avg = [labels_avg] * N
    if titles_1d is None:
        titles_1d = childkeys_1d
    if labels_1d is None:
        labels_1d = childkeys_1d
    if not isinstance(titles_1d, list):
        titles_1d = [titles_1d] * N
    if not isinstance(labels_1d, list):
        labels_1d = [labels_1d] * N

    # Slider run part
    # one slider to control *number of runs* only
    slider_runs = Slider(
        start=1, end=file[parentkeys_avg[0]][childkeys_avg[0]].shape[0],
        value=max(1, int(n_runs)), step=1, title="Number of runs to average"
    )
    WIDGETS["hist_runs"] = slider_runs
    displays_avg = [i[0] for i in histogram_avg_layout.children]

    # Slider bin part
    slider_bins = Slider(
        start=2, end=100,
        value=max(2, int(n_bins)), step=1, title="Number of bins"
    )
    WIDGETS["hist_bins"] = slider_bins

    # callback: recompute all displays' histograms using current slider value
    def _on_runs_change(attr, old, new):
        current_runs = slider_runs.value
        n_bins = slider_bins.value
        for index in range(len(displays_avg)):
            data = file[parentkeys_avg[index]][childkeys_avg[index]]
            _recompute_display_hist(
                displays_avg[index], data, labels_avg[index], current_runs, n_bins
            )

    slider_runs.on_change("value", _on_runs_change)

    # initial recompute (ensures sources are consistent)
    _on_runs_change(None, None, None)

    # Div suptitle
    if suptitle_avg is None:
        suptitle_avg = "<strong>Subsection</strong>"
    else:
        suptitle_avg = "<strong>" + suptitle_avg + "</strong>"
    name = Div(text=suptitle_avg)

    display_layout = column(name, slider_runs, histogram_avg_layout)

    # callback: recompute all displays' histograms using current slider value
    def _on_bins_change(attr, old, new):
        current_bins = slider_bins.value
        current_runs = slider_runs.value
        for index in range(len(displays_avg)):
            data = file[parentkeys_avg[index]][childkeys_avg[index]]
            _recompute_display_hist(
                displays_avg[index],
                data,
                labels_avg[index],
                n_runs=current_runs,
                n_bins=current_bins
            )
        for index in range(len(histogram_1d_layout.children[1].children)):
            data = file[parentkeys_1d[index]][childkeys_1d[index]]
            _recompute_display_hist_for_1d(
                histogram_1d_layout.children[1].children[index][0],
                data,
                labels_1d[index],
                n_bins=current_bins
            )

    slider_bins.on_change("value", _on_bins_change)

    # initial recompute (ensures sources are consistent)
    _on_bins_change(None, None, None)
    
    histogram_layout = column(slider_bins, display_layout, histogram_1d_layout)
    return histogram_layout

def make_annulus(
    file,
    childkey,
    parentkey,
    title=None
):
    if title is None:
        title = childkey

    display = BokehPlot(
        title=title,
        tools=("xpan", "box_zoom", "wheel_zoom", "save", "reset"),
        active_drag="xpan",
        toolbar_location=None
    )
    xdr = Range1d(start=-2, end=2)
    ydr = Range1d(start=-2, end=2)
    display.figure = Plot(x_range=xdr, y_range=ydr)
    fig = display.figure
    fig.title.text = title
        
    data = np.asarray(file[parentkey][childkey])
    group, counts = np.unique(data, return_counts=True)
    angles = np.concatenate(([0], 2 * np.pi * np.cumsum(counts) / np.sum(counts)))
    source = ColumnDataSource(
        {
            "start": angles[:-1],
            "end": angles[1:],
            "colors": Inferno[len(group)+2][1:-1],
            "counts": counts
        }
    )
    glyph = AnnularWedge(
        x=0, y=0, inner_radius=0.9, outer_radius=1.8,
        start_angle="start", end_angle="end",
        line_color="white", line_width=3, fill_color="colors"
    )
    r = fig.add_glyph(source, glyph)

    hover = HoverTool(
        tooltips=[("Count", "@{}".format("counts"))],
        renderers=[r]
    )
    fig.add_tools(hover)
    
    legend = Legend(location="center")
    for i, name in enumerate(group):
        legend.items.append(LegendItem(label=str(name), renderers=[r], index=i))
    fig.add_layout(legend, "center")

    display._meta = {
        "type": "annulus",
        "parentkey": parentkey,
        "childkey": childkey,
        "factory": "make_annulus"
    }
    DISPLAY_REGISTRY.append(display)
    
    return display

def make_annulii(
    file,
    childkeys,
    parentkeys,
    titles=None,
    suptitle=None
):
    N = len(childkeys)
    if not isinstance(parentkeys, list):
        parentkeys = [parentkeys] * N
    if titles is None:
        titles = childkeys
    if not isinstance(titles, list):
        titles = [titles] * N

    displays = []
    for index in range(len(childkeys)):
        displays.append(
            make_annulus(
                file,
                childkeys[index],
                parentkeys[index],
                titles[index]
            )
        )

    if suptitle is None:
        suptitle = "<strong>Subsection</strong>"
    name = Div(text=suptitle)

    display_gridplot = gridplot([d.figure for d in displays], ncols=2)
    display_layout = column(name, display_gridplot)
    return display_layout

def make_full_histogram_sections(
    file,
    childkeys_avg,
    parentkeys_avg,
    childkeys_1d,
    parentkeys_1d,
    childkeys_pie,
    parentkeys_pie,
    n_runs=1,
    n_bins=50,
    titles_avg=None,
    labels_avg=None,
    suptitle_avg=None,
    titles_1d=None,
    labels_1d=None,
    suptitle_1d=None,
    titles_pie=None,
    suptitle_pie=None
):
    histogram_layout = make_histogram_sections(
        file,
        childkeys_avg,
        parentkeys_avg,
        childkeys_1d,
        parentkeys_1d,
        n_runs=n_runs,
        n_bins=n_bins,
        titles_avg=titles_avg,
        labels_avg=labels_avg,
        suptitle_avg=suptitle_avg,
        titles_1d=titles_1d,
        labels_1d=labels_1d,
        suptitle_1d=suptitle_1d
    )
    annulii_layout = make_annulii(
        file,
        childkeys_pie,
        parentkeys_pie,
        titles=titles_pie,
        suptitle=suptitle_pie
    )
    full_histogram_layout = column(histogram_layout, annulii_layout)
    return TabPanel(child=full_histogram_layout, title="Histograms")

def make_body(
    file,
    childkeys_camera=["image", "peak_time"],
    parentkeys_camera="dl1/event/telescope/images/tel_001",
    childkeys_parameter=[
        "hillas_x", "hillas_y",
        "hillas_phi", "hillas_width",
        "hillas_length", "hillas_intensity",
        "total_intensity",
        "hillas_r", "hillas_psi",
        "hillas_skewness", "hillas_kurtosis"
    ],
    parentkeys_parameter="dl1/event/telescope/parameters/tel_001",
    childkeys_monitoring=["event_type", "is_good_event", "event_quality"],
    parentkeys_monitoring="dl1/event/telescope/parameters/tel_001",
    label_2d_timeline=["Min", "Mean", "Median", "Max"],
    func_timeline=[np.min, np.mean, np.median, np.max],
    ylabel_2d=["Image [p.e.]", "Peak time [ns]"],
    ylabel_1d=[
        "Hillas: x [unit]",
        "Hillas: y [unit]",
        "Hillas: &#966; [unit]",
        "Hillas: width [unit]",
        "Hillas: length [unit]",
        "Hillas: maximum intensity [unit]",
        "Hillas: total intensity [unit]",
        "Hillas: radius [unit]",
        "Hillas: &#968; [unit]",
        "Hillas: skewness [unit]",
        "Hillas: kurtosis [unit]",
    ],
    ylabel_step=[
        "Event type",
        "Is good event",
        "Event quality"
    ],
    suptitle_2d="Raw camera data",
    suptitle_1d="Hillas parameters",
    suptitle_step="Event monitoring",
    n_runs=default_n_runs,
    n_bins=default_n_bins,
    labels_colorbar=["p.e.", "ns"]
):
    
    # Summary card
    summary_card = make_summary_card(file)
    
    # Camera displays
    tab_camera_displays = make_tab_camera_displays(
        file,
        childkeys_camera,
        labels_colorbar=labels_colorbar
    )
    
    # Timelines
    tab_timelines = make_tab_timelines(
        file,
        childkeys_camera,
        parentkeys_camera,
        childkeys_parameter,
        parentkeys_parameter,
        childkeys_monitoring,
        parentkeys_monitoring,
        labels_2d=label_2d_timeline,
        funcs=func_timeline,
        suptitle_2d=suptitle_2d,
        suptitle_1d=suptitle_1d,
        suptitle_step=suptitle_step,
        #ylabel_2d=ylabel_2d,
        #ylabel_1d=ylabel_1d,
        #ylabel_step=ylabel_step,
    )
    
    # Histograms
    tab_histograms = make_full_histogram_sections(
        file,
        childkeys_camera,
        parentkeys_camera,
        childkeys_parameter,
        parentkeys_parameter,
        childkeys_monitoring,
        parentkeys_monitoring,
        n_runs=n_runs,
        n_bins=n_bins,
        suptitle_avg=suptitle_2d,
        suptitle_1d=suptitle_1d,
        suptitle_pie=suptitle_step
    )
    
    # Skymaps
    skymaps = Div(text="TBD - Waiting for DL2 & DL3")
    tab_skymaps = TabPanel(child=skymaps, title="Skymaps")
    
    # Tabs
    tabs = Tabs(
        tabs=[tab_camera_displays, tab_timelines, tab_histograms, tab_skymaps],
    )
    
    # Build layout
    return row(summary_card, tabs)