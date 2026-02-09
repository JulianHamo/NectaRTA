# Author: Julian Hamo
# Contact mail: julian.hamo@ijclab.in2p3.fr

"""
This module stores the Bokeh webpage utility helpers for the RTA of NectarCAM.
"""


__all__ = ["get_hillas_parameters"]


def get_hillas_parameters(
    file,
    parameterkeys,
    parameter_parentkeys,
    run_index=-1
):
    """Get the Hillas parameters from the file.

    Parameters
    ----------
    file : hdf5 file
        File to retrieve the data.
    parameterkeys : dict
        Dictionnary of parameter keys to retrieve
        the Hillas parameters from ``file``.
    parameter_parentkeys: string
        Parent key for the parameters in the dictionary
    run_index : int, optional
        A file is constituted of multiple events,
        select an event in the stored file.
        Default is -1, resulting in the latest event of the run.

    Returns
    -------
    x : float
        Position of the center of the Hillas ellipse on the x axis.
    y : float
        Position of the center of the Hillas ellipse on the y axis.
    width : float
        Width of the Hillas ellipse.
    length : float
        Length of the Hillas ellipse.
    angle : float
        Angle between the mahor axis of the Hillas ellipse and the x axis.
        
    """

    try:
        x=file[parameter_parentkeys][parameterkeys["hillas_x_key"]][run_index]
        y=file[parameter_parentkeys][parameterkeys["hillas_y_key"]][run_index]
        width=file[parameter_parentkeys][parameterkeys["hillas_length_key"]][run_index]
        height=file[parameter_parentkeys][parameterkeys["hillas_width_key"]][run_index]
        angle=file[parameter_parentkeys][parameterkeys["hillas_phi_key"]][run_index]
        return x, y, width, height, angle
    except Exception as e:
        print("Failed to retrieve Hillas parameters:", e)