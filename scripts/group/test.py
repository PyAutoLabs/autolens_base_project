from astropy.io import fits
from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()
from os import path
import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load, plot and mask the `Imaging` data.
"""
#dataset_name = "102021990_NEG650312660474055399"
dataset_name = "102019596_NEG637748088500119037"
dataset_path = path.join("..", "..", "dataset", "sample_group", dataset_name)
dataset_path = Path(dataset_path)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
# if not dataset_path.exists():
#     import subprocess
#     import sys
#
#     subprocess.run(
#         [sys.executable, "scripts/group/simulator.py"],
#         check=True,
#     )

pixel_scale = 0.1
mask_radius = 3.0
mask_centre = (0.0, 0.0)
redshift_lens = 0.5
redshift_source = 1.0
source_mge_radius = 1.0
n_batch = 20

from astropy.io import fits

def dataset_instrument_hdu_dict_via_fits_from(
    dataset_path, dataset_fits_name, image_tag: str = "_FLUX"
):
    """
    Load a dictionary mapping dataset instruments (e.g. DES_g, NIR_Y) to their index in a multi-extension
    fits file.

    Parameters
    ----------
    dataset_path
        The path where the multi-extension fits file is stored.
    dataset_fits_name
        The name of the multi-extension fits file.
    image_tag
        The tag appended to the instrument name of the image HDU, e.g. _FLUX, _IMAGE, which is used to pick
        out the image HDUs from the fits file and ignore other HDUs like noise maps or PSFs.

    Returns
    -------
    A dictionary mapping dataset names to their index in the fits file.
    """
    hdu_list = fits.open(dataset_path / dataset_fits_name)

    # Build dictionary: {name: index}
    hdu_dict = {}
    for i, hdu in enumerate(hdu_list):
        name = hdu.name if hdu.name else ("PRIMARY" if i == 0 else f"UNNAMED_{i}")
        hdu_dict[name] = i

    instrument_dict = {}
    counter = 0

    for hdu in hdu_list:
        name = hdu.name
        if name.endswith(image_tag):
            band = name.replace(image_tag, "").lower()
            instrument_dict[band] = counter
            counter += 1

    return instrument_dict

hdu_list = fits.open(dataset_path / 'data.fits')
print(hdu_list[1].header)
# Build dictionary: {name: index}
hdu_dict = {}
for i, hdu in enumerate(hdu_list):
    name = hdu.name if hdu.name else ("PRIMARY" if i == 0 else f"UNNAMED_{i}")
    hdu_dict[name] = i

print(name)
print(hdu_dict)

dataset_index_dict = dataset_instrument_hdu_dict_via_fits_from(
    dataset_path=dataset_path,
    dataset_fits_name="data.fits",
    image_tag="_FLUX",
)

print(dataset_index_dict)

vis_index = dataset_index_dict["vis"]
print(vis_index)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    data_hdu=vis_index * 3 + 1,
    noise_map_path=dataset_path / "data.fits",
    noise_map_hdu=vis_index * 3 + 3,
    psf_path=dataset_path / "data.fits",
    psf_hdu=vis_index * 3 + 2,
    pixel_scales=pixel_scale,
    check_noise_map=False,
)

import autolens as al
import autolens.plot as aplt

aplt.subplot_imaging_dataset(dataset=dataset)