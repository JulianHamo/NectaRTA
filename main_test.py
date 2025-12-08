# main.py
"""
Main Bokeh server app entry point.

Assumes:
 - display.py has been updated to export:
     DISPLAY_REGISTRY  (list)
     WIDGETS           (dict)
     make_header_menu(...)
     make_body(...)    (or a function that builds the body/layout)
 - factories in display.py attach display._meta dict describing how to update them
   (example keys: 'type', 'parentkey', 'childkey', 'label', etc)
 - utils.data_download.get_latest_file(RESSOURCE_PATH) returns an open h5py-like file
"""
import os
import time
import traceback

import h5py
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import row, column, Column
from bokeh.models import Select, Div
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

# Import what we expect from your display module
from display import (
get_latest_file,
    make_header_menu,
    make_body,
    DISPLAY_REGISTRY,
    WIDGETS,
)

# ----------------------------
# Configuration
# ----------------------------
RESSOURCE_PATH = "../../RTA_output_stream/output_RTA_run_restreamed_20250707"
REAL_TIME_TAG = "Real time"   # choose the exact string your header uses
DEFAULT_UPDATE_MS = 3000      # milliseconds between updates in real-time mode

# Globals
CURRENT_FILE = None           # open h5py.File-like object used by update_figures
CURRENT_FILE_PATH = None
PERIODIC_CB_ID = None
STATUS_DIV = None             # optional status element returned from header
HEADER_SELECT = None          # Select widget for header (value used to choose file)
ROOT_LAYOUT = None            # top-level layout (column)
# ----------------------------


# ----------------------------
# Utilities: helpers for updating displays
# ----------------------------
def safe_close_file(fobj):
    try:
        if fobj is not None:
            # h5py File has .close(); handle other types defensively
            if hasattr(fobj, "close"):
                fobj.close()
    except Exception:
        pass


def open_file_for_selection(sel_value):
    """Return an open h5py.File-like object for selection.
    sel_value == REAL_TIME_TAG -> returns get_latest_file(RESSOURCE_PATH)
    else -> expects sel_value to be a filename (without path) or full path.
    """
    if sel_value is None:
        return None, None

    if sel_value == REAL_TIME_TAG:
        f = get_latest_file(RESSOURCE_PATH)
        # try to get a path string for display
        path = getattr(f, "filename", None)
        return f, path

    # selection might be a displayed name; attempt to build a path
    # try many reasonable variants
    candidates = [
        sel_value,
        os.path.join(RESSOURCE_PATH, sel_value),
        os.path.join(RESSOURCE_PATH, sel_value + ".h5"),
        os.path.join(RESSOURCE_PATH, sel_value + ".hdf5"),
    ]
    for c in candidates:
        try:
            if c is None:
                continue
            if os.path.exists(c):
                f = h5py.File(c, "r")
                return f, c
        except Exception:
            continue
    # if nothing found, try letting your get_latest_file find something named sel_value
    try:
        f = get_latest_file(RESSOURCE_PATH)
        return f, getattr(f, "filename", None)
    except Exception:
        return None, None


def _get_vbar_source_from_figure(fig):
    """Try to find the ColumnDataSource used by a vbar renderer in fig.
    Returns the first found ColumnDataSource or None.
    """
    for r in getattr(fig, "renderers", []):
        ds = getattr(r, "data_source", None)
        # Many glyph renderers use ColumnDataSource; we return the first that looks like histogram source
        if isinstance(ds, ColumnDataSource):
            # heuristic: must contain two columns typical of hist: edges or counts
            keys = set(ds.data.keys())
            if {"edges", "counts"}.intersection(keys) or {"edges"}.issubset(keys):
                return ds
            # fallback: return any CDS
            return ds
    return None


def _recompute_hist_and_update_source(disp, parentkey, childkey, label, n_runs, n_bins):
    """Read the dataset from CURRENT_FILE and recompute histogram, then update display's source or figure."""
    global CURRENT_FILE
    try:
        if CURRENT_FILE is None:
            return

        # read dataset robustly
        try:
            arr = np.asarray(CURRENT_FILE[parentkey][childkey])
        except Exception as e:
            # dataset missing or unreadable => nothing to update
            print(f"_recompute_hist_and_update_source: failed read {parentkey}/{childkey}: {e}")
            return

        # normalize shape and build flattened sample from last n_runs
        if arr.ndim == 0:
            sample = np.array([arr])
        elif arr.ndim == 1:
            n_runs_use = max(1, min(int(n_runs), arr.shape[0]))
            sample = arr[-n_runs_use:]
        else:
            n_runs_use = max(1, min(int(n_runs), arr.shape[0]))
            sample = arr[-n_runs_use:].ravel()

        n_bins = max(1, int(n_bins))
        hist, edges = np.histogram(sample, bins=n_bins)
        centers = (edges[:-1] + edges[1:]) / 2.0

        # Attempt in-place update:
        # 1) If display has attribute 'source', use it
        if hasattr(disp, "source") and isinstance(disp.source, ColumnDataSource):
            # update with conventional keys if available
            try:
                # preserve column name if label used
                disp.source.data = {label: hist.astype(int), "edges": centers}
                # try to update glyph width if we can find renderer
                for r in getattr(disp.figure, "renderers", []):
                    if hasattr(r.glyph, "width"):
                        r.glyph.width = 0.9 * (edges[1] - edges[0]) if len(edges) > 1 else 1.0
                return
            except Exception as e:
                print("_recompute_hist_and_update_source: error updating disp.source:", e)

        # 2) Attempt to locate a vbar source inside the figure
        ds = _get_vbar_source_from_figure(disp.figure)
        if isinstance(ds, ColumnDataSource):
            try:
                # heuristics: some code expects keys 'counts' and 'edges' or label and 'edges'
                if "counts" in ds.data:
                    ds.data = dict(counts=hist.astype(int), edges=centers)
                else:
                    ds.data = {label: hist.astype(int), "edges": centers}
                # update glyph widths
                for r in getattr(disp.figure, "renderers", []):
                    if hasattr(r.glyph, "width"):
                        r.glyph.width = 0.9 * (edges[1] - edges[0]) if len(edges) > 1 else 1.0
                return
            except Exception as e:
                print("_recompute_hist_and_update_source: error updating located CDS:", e)

        # 3) Fallback: replace the figure entirely by redrawing a simple histogram into disp.figure
        try:
            disp.figure.renderers = []  # clear old glyphs
            p = disp.figure
            width = (edges[1] - edges[0]) if len(edges) > 1 else 1.0
            p.vbar(x=centers, top=hist, width=0.9 * width, line_color="navy", fill_color=None)
            p.x_range.start = centers.min() if len(centers) else 0
            p.x_range.end = centers.max() if len(centers) else 1
            p.y_range.start = 0
            return
        except Exception as e:
            print("_recompute_hist_and_update_source: fallback redraw failed:", e)
            return

    except Exception as e:
        print("_recompute_hist_and_update_source: unexpected error:", e)
        traceback.print_exc()


def _recompute_camera_display(disp, parentkey, childkey, run_index):
    """Update camera display's .image if available."""
    global CURRENT_FILE
    if CURRENT_FILE is None:
        return
    try:
        imgds = CURRENT_FILE[parentkey][childkey]
        # try to index safely
        try:
            img = np.nan_to_num(np.asarray(imgds[run_index]), nan=0.0)
        except Exception:
            # fallback to last event
            img = np.nan_to_num(np.asarray(imgds[-1]), nan=0.0)
        # if disp has attribute image, set it
        try:
            disp.image = img
            # Some CameraDisplay objects require calling add_colorbar or refresh; ignore
            return
        except Exception:
            # try to find a glyph or source in the figure and replace it (not ideal)
            try:
                ds = _get_vbar_source_from_figure(disp.figure)
                if isinstance(ds, ColumnDataSource):
                    # replace with flattened pixel values as naive fallback
                    ds.data = {"vals": img.ravel(), "idx": np.arange(len(img.ravel()))}
            except Exception:
                pass
    except Exception as e:
        print("_recompute_camera_display: failed", e)


def _recompute_timeline_display(disp, parentkey, childkey, meta):
    """Attempt to recompute timeline series and update any ColumnDataSource found in the figure."""
    global CURRENT_FILE
    if CURRENT_FILE is None:
        print("No file found: empty return")
        return
    try:
        arr = np.asarray(CURRENT_FILE[parentkey][childkey])
    except Exception as e:
        print("_recompute_timeline_display: read failed:", e)
        return

    # compute standard series if they are typical
    try:
        mean = np.nanmean(arr, axis=-1)
        median = np.nanmedian(arr, axis=-1)
        mx = np.nanmax(arr, axis=-1)
        mn = np.nanmin(arr, axis=-1)
    except Exception:
        print("Could not compute any statistics: empty timelines")
        mean = np.array([])
        median = np.array([])
        mx = np.array([])
        mn = np.array([])

    # update any CDS that contains matching keys
    for r in getattr(disp.figure, "renderers", []):
        src = getattr(r, "data_source", None)
        if not isinstance(src, ColumnDataSource):
            continue
        # try to update common column names
        data_keys = src.data.keys()
        update_dict = {
            "time": np.arange(arr.shape[0])        }
        if np.all(np.isin(list(data_keys), ["time", "Min", "Median", "Mean", "Max"])):
            update_dict |= {
                "Min": mn,
                "Mean": mean,
                "Median": median,
                "Max": mx
            }
        elif np.all(np.isin(list(data_keys), ["time", "y"])):
            update_dict |= {
                "y": arr
            }
        else:
            print("No right data format found")
            return
        src.data = update_dict


# ----------------------------
# High-level update loop
# ----------------------------
def update_figures():
    """Update all displays in DISPLAY_REGISTRY according to their ._meta using CURRENT_FILE and WIDGETS.
       Only figures are updated (widgets are left untouched)."""
    global DISPLAY_REGISTRY, CURRENT_FILE

    if not DISPLAY_REGISTRY:
        return

    for disp in DISPLAY_REGISTRY:
        meta = getattr(disp, "_meta", None)
        if not meta:
            continue

        try:
            dtype = meta.get("type", "").lower()

            if dtype == "hist_avg" or dtype == "hist_1d" or dtype.startswith("hist"):
                parent = meta.get("parentkey")
                child = meta.get("childkey")
                label = meta.get("label", meta.get("childkey", "value"))
                # read widget values (fall back to meta defaults)
                n_runs_widget = WIDGETS.get("hist_runs")
                n_bins_widget = WIDGETS.get("hist_bins")
                n_runs = int(n_runs_widget.value) if getattr(n_runs_widget, "value", None) is not None else meta.get("n_runs", 1)
                n_bins = int(n_bins_widget.value) if getattr(n_bins_widget, "value", None) is not None else meta.get("n_bins", 50)

                _recompute_hist_and_update_source(disp, parent, child, label, n_runs, n_bins)

            elif dtype == "camera":
                # choose run index from widget if present
                run_widget = WIDGETS.get("camera_run")
                run_index = int(run_widget.value) if getattr(run_widget, "value", None) is not None else -1
                parent = meta.get("image_parentkey") or meta.get("parentkey")
                child = meta.get("childkey")
                _recompute_camera_display(disp, parent, child, run_index)

            elif dtype.startswith("timeline"):
                parent = meta.get("parentkey")
                child = meta.get("childkey")
                _recompute_timeline_display(disp, parent, child, meta)

            else:
                # Unknown type: try generic attempts: if disp has .source and meta defines parent/child/label, try histogram-like update
                parent = meta.get("parentkey")
                child = meta.get("childkey")
                label = meta.get("label", None)
                if parent and child and label:
                    n_runs_widget = WIDGETS.get("hist_runs")
                    n_bins_widget = WIDGETS.get("hist_bins")
                    n_runs = int(n_runs_widget.value) if getattr(n_runs_widget, "value", None) is not None else meta.get("n_runs", 1)
                    n_bins = int(n_bins_widget.value) if getattr(n_bins_widget, "value", None) is not None else meta.get("n_bins", 50)
                    _recompute_hist_and_update_source(disp, parent, child, label, n_runs, n_bins)
                # else: nothing we can reliably update generically

        except Exception as e:
            print("update_figures: update failed for display meta=", meta, "error=", e)
            traceback.print_exc()

def update_timestamp():
    ts = time.strftime('%H:%M:%S')
    print(f"Real time mode: updating figure - {ts}")
    STATUS_DIV.children[1].text = f"Last update: {ts}"

def periodic_update():
    update_figures()
    update_timestamp()


# ----------------------------
# Periodic callback management
# ----------------------------
def start_periodic_updates(interval_ms=DEFAULT_UPDATE_MS):
    global PERIODIC_CB_ID
    if PERIODIC_CB_ID is not None:
        return
    PERIODIC_CB_ID = curdoc().add_periodic_callback(periodic_update, interval_ms)
    print(f"Periodic updates started (id={PERIODIC_CB_ID}, interval_ms={interval_ms})")


def stop_periodic_updates():
    global PERIODIC_CB_ID
    if PERIODIC_CB_ID is None:
        return
    try:
        curdoc().remove_periodic_callback(PERIODIC_CB_ID)
    except Exception:
        pass
    PERIODIC_CB_ID = None
    print("Periodic updates stopped")


# ----------------------------
# Header select callback
# ----------------------------
def _on_header_select_change(attr, old, new):
    """Called when header selector changes value."""
    global CURRENT_FILE, CURRENT_FILE_PATH

    sel = new
    # stop any real-time periodic update
    stop_periodic_updates()

    # Close previously opened non-latest file (if any)
    try:
        if CURRENT_FILE is not None and CURRENT_FILE_PATH is not None and CURRENT_FILE_PATH != getattr(CURRENT_FILE, "filename", None):
            # if it was opened by us with h5py, close it
            safe_close_file(CURRENT_FILE)
    except Exception:
        pass

    # Decide what to do
    if sel == REAL_TIME_TAG:
        # use latest file and start periodic updates
        try:
            CURRENT_FILE, CURRENT_FILE_PATH = open_file_for_selection(REAL_TIME_TAG)
            # run an initial update using current widget values
            update_figures()
            # start periodic updates
            start_periodic_updates(DEFAULT_UPDATE_MS)
            _set_status_text(f"In real-time mode (refresh every {DEFAULT_UPDATE_MS} ms). Using: {CURRENT_FILE_PATH}")
        except Exception as e:
            _set_status_text(f"Failed to start real-time mode: {e}")
            print("Failed to start real-time mode:", e)
    else:
        # open the selected file and update once (no periodic updates)
        try:
            fobj, fpath = open_file_for_selection(sel)
            if fobj is None:
                _set_status_text(f"Could not open selected file: {sel}")
                return
            CURRENT_FILE = fobj
            CURRENT_FILE_PATH = fpath
            update_figures()
            _set_status_text(f"Loaded file: {CURRENT_FILE_PATH} (static mode)")
        except Exception as e:
            _set_status_text(f"Could not open selected file: {sel} : {e}")
            print("Could not open selected file:", e)


def _set_status_text(msg):
    """Helper to set status in header if STATUS_DIV exists."""
    try:
        global STATUS_DIV
        if STATUS_DIV is None:
            return
        # STATUS_DIV may be a Div or container; attempt to set .text or .children
        if isinstance(STATUS_DIV, Div):
            STATUS_DIV.text = msg
        else:
            # if it's a layout, attempt to set first child text
            try:
                if hasattr(STATUS_DIV, "children") and len(STATUS_DIV.children) > 0:
                    child = STATUS_DIV.children[0]
                    if isinstance(child, Div):
                        child.text = msg
            except Exception:
                pass
    except Exception:
        pass


# ----------------------------
# Build UI: header + body
# ----------------------------
def build_ui():
    global HEADER_SELECT, STATUS_DIV, ROOT_LAYOUT, CURRENT_FILE, CURRENT_FILE_PATH

    # call make_header_menu; be defensive about its return signature
    header_ret = make_header_menu(RESSOURCE_PATH)
    # possible return shapes handled:
    # - a Select widget alone
    # - (select_widget, status_div)
    # - (container_row, file)
    sel_widget = None
    status_obj = None

    if isinstance(header_ret, tuple) or isinstance(header_ret, list):
        # try to find a Select inside the tuple or assume first is select
        # common pattern: (select_widget, status_div)
        for item in header_ret:
            if isinstance(item, Select):
                sel_widget = item
            elif isinstance(item, Column):
                status_obj = item
        # if not found, assume header_ret[0] is select-like
        if sel_widget is None and len(header_ret) > 0 and hasattr(header_ret[0], "on_change"):
            sel_widget = header_ret[0]
        if status_obj is None and len(header_ret) > 1 and isinstance(header_ret[1], Column):
            status_obj = header_ret[1]
    else:
        # single return: maybe it's the select widget or a layout containing it
        if isinstance(header_ret, Select):
            sel_widget = header_ret
        else:
            # assume it's a layout; try to find Select inside by attribute inspection
            try:
                # if header_ret has children, search them
                children = getattr(header_ret, "children", [])
                for c in children:
                    if isinstance(c, Select):
                        sel_widget = c
                    elif isinstance(c, Column):
                        status_obj = c
            except Exception:
                pass

    HEADER_SELECT = sel_widget
    STATUS_DIV = status_obj

    # If no select found, create a simple select as fallback
    if HEADER_SELECT is None:
        HEADER_SELECT = Select(title="Run", value=REAL_TIME_TAG, options=[REAL_TIME_TAG])
        STATUS_DIV = Div(text="(no header provided)", width=600)

    # create initial file: either latest or nothing
    try:
        CURRENT_FILE, CURRENT_FILE_PATH = open_file_for_selection(HEADER_SELECT.value if hasattr(HEADER_SELECT, "value") else REAL_TIME_TAG)
    except Exception:
        CURRENT_FILE, CURRENT_FILE_PATH = None, None

    # Build body (try passing the file if make_body accepts it)
    body_ret = None
    try:
        # attempt call with file
        body_ret = make_body(CURRENT_FILE)
    except TypeError:
        try:
            # fallback: call without parameters
            body_ret = make_body()
        except Exception as e:
            print("make_body failed:", e)
            body_ret = Div(text="Error building body")

    # assemble root layout
    ROOT_LAYOUT = column(
        row(header_ret[0], header_ret[1]) if header_ret is not None else row(HEADER_SELECT, STATUS_DIV),
        body_ret,
        sizing_mode="scale_width"
        )
    curdoc().add_root(ROOT_LAYOUT)

    # wire select callback
    try:
        HEADER_SELECT.on_change("value", _on_header_select_change)
    except Exception:
        # maybe header_ret included its own wiring; still attach if possible
        pass

    # set status if available
    if STATUS_DIV is not None:
        _set_status_text(f"Loaded: {CURRENT_FILE_PATH}")


# Build UI now
build_ui()

# Optionally start real-time at launch if header default equals REAL_TIME_TAG
try:
    if getattr(HEADER_SELECT, "value", None) == REAL_TIME_TAG:
        # perform an initial update and start periodic updates
        update_figures()
        start_periodic_updates(DEFAULT_UPDATE_MS)
except Exception:
    pass