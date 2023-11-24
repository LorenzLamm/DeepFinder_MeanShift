import numpy as np 
import mrcfile
from skimage.util import img_as_float32

def store_tomogram(
    filename: str, tomogram, voxel_size=None
) -> None:
    """
    Store tomogram in specified path.

    Parameters
    ----------
    filename : str
        Name of the file to store the tomogram.
    tomogram : Tomogram or np.ndarray
        The tomogram data if given as np.ndarray. If given as a Tomogram,
        both data and header are used for storing.
    voxel_size: float, optional
        If specified, this voxel size will be stored into the tomogram header.
    """
    with mrcfile.new(filename, overwrite=True) as out_mrc:
        data = tomogram
        data = np.transpose(data, (2, 1, 0))
        out_mrc.set_data(data)
        if voxel_size is not None:
            out_mrc.voxel_size = voxel_size


def load_tomogram(
    filename: str,
    normalize_data: bool = False,
) -> np.ndarray:
    """
    Loads tomogram and transposes s.t. we have data in the form x,y,z.

    If specified, also normalizes the tomogram.

    Parameters
    ----------
    filename : str
        File name of the tomogram to load.
    normalize_data : bool, optional
        If True, normalize data.

    Returns
    -------
    tomogram : Tomogram
        A Tomogram dataclass containing the loaded data, header
        and voxel size.

    """
    with mrcfile.open(filename, permissive=True) as tomogram:
        data = tomogram.data.copy()
        data = np.transpose(data, (2, 1, 0))
    if normalize_data:
        data = img_as_float32(data)
        data -= np.mean(data)
        data /= np.std(data)
    return data