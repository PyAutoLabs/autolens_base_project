"""
SLaM (Source, Light and Mass): Group — Multiple Main Lenses
============================================================

This script extends the group pipeline to support multiple main lens galaxies
(e.g. a merging pair or a group-core triplet). The number of main lenses is
determined at runtime from ``main_lens_centres.json``.

Galaxy populations
------------------
- **Main lenses** (``lens_0``, ``lens_1``, ...): each has an MGE bulge, an
  Isothermal mass, and ExternalShear. Centres are pinned to the positions in
  ``main_lens_centres.json``.
- **Extra lens galaxies**: nearby companions with individual MGE light + Isothermal
  mass. Loaded from ``extra_galaxies_centres.json`` (optional).
- **Scaling lens galaxies**: more distant galaxies modeled with a shared
  luminosity scaling relation. Loaded from ``scaling_galaxies_centres.json``
  (optional).

Pipeline stages
---------------
source_lp[0]   run_source_lp_0   — light only, no mass, no source
source_lp[1]   run_source_lp_1   — full LP fit, mass + source introduced
source_pix[1]  run_source_pix_1  — pixelized source, free Isothermal mass
source_pix[2]  run_source_pix_2  — refined pixelization, all lens components fixed
light[1]       run_light_lp      — free MGE light, fixed mass + source
mass_total[1]  run_mass_total    — free PowerLaw mass, fixed light + source
"""

import numpy as np
import json
from pathlib import Path

import autofit as af
import autolens as al


def _load_centres(path):
    """Load a centres JSON file, returning an empty list if the file is absent."""
    try:
        return al.Grid2DIrregular(al.from_json(file_path=path))
    except FileNotFoundError:
        return al.Grid2DIrregular([])


def run_source_lp_0(
    dataset,
    settings_search,
    main_lens_centres,
    extra_lens_centres,
    scaling_lens_centres,
    mask_radius,
    redshift_lens,
    n_batch=50,
):
    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the SOURCE LP PIPELINE fits a lens model where:

     - Every main lens galaxy is modeled using a free MGE light profile [no prior initialization].
     - Every extra galaxy is modeled using a free MGE light profile [no prior initialization].
     - Every scaling galaxy is modeled using a free MGE light profile [no prior initialization].

    This search aims to accurately estimate the light distribution of all galaxies before the
    mass model and source are introduced.
    """
    analysis = al.AnalysisImaging(dataset=dataset)

    # --- main lens light models (one per centre, light only) ---
    lens_light_models = []
    for centre in main_lens_centres:
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=30,
            gaussian_per_basis=2,
            centre_prior_is_uniform=False,
            centre=(centre[0], centre[1]),
            centre_sigma=0.1,
        )
        lens_light_models.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge, disk=None, point=None)
        )

    # --- extra lens galaxy light models ---
    extra_lens_light_list = []
    for centre in extra_lens_centres:
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=10,
            centre_prior_is_uniform=False,
            centre=(centre[0], centre[1]),
            centre_sigma=0.1,
        )
        extra_lens_light_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge)
        )

    extra_galaxies = af.Collection(extra_lens_light_list) if extra_lens_light_list else None

    # --- scaling galaxy light models ---
    scaling_light_list = []
    for centre in scaling_lens_centres:
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=10,
            centre_prior_is_uniform=False,
            centre=(centre[0], centre[1]),
            centre_sigma=0.1,
        )
        scaling_light_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge)
        )

    scaling_galaxies = af.Collection(scaling_light_list) if scaling_light_list else None

    n_extra = len(extra_galaxies) if extra_galaxies is not None else 0
    n_scaling = len(scaling_galaxies) if scaling_galaxies is not None else 0
    n_live = 100 + 30 * len(lens_light_models) + 30 * n_extra + 30 * n_scaling

    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_light_models)}
    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
        scaling_galaxies=scaling_galaxies,
    )

    search = af.Nautilus(
        name="source_lp[0]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
        n_like_max=1000000,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def run_source_lp_1(
    dataset,
    settings_search,
    source_lp_result_0,
    positions,
    pixel_scale,
    redshift_lens,
    redshift_source,
    source_mge_radius,
    n_batch=50,
):
    """
    __Model + Search + Analysis + Model-Fit (Search 2)__

    Search 2 of the SOURCE LP PIPELINE fits a lens model where:

     - Every main lens galaxy has light fixed from source_lp[0] and a free Isothermal
       mass (centre pinned to the stage-0 bulge centre, broad Einstein radius prior).
       ExternalShear is free on lens_0 only.
     - Every extra galaxy has light fixed from source_lp[0] and a free Isothermal mass
       bounded by the stage-0 luminosity.
     - Every scaling galaxy has light fixed from source_lp[0] and mass set by a shared
       free luminosity scaling relation.
     - The source galaxy is a free MGE light profile.

    This search aims to produce an initial joint lens plus source model with a
    physically motivated mass scaling for companion galaxies.
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[al.PositionsLH(positions=positions, threshold=0.3)],
    )

    n_main = sum(
        1 for k in vars(source_lp_result_0.instance.galaxies) if k.startswith("lens_")
    )
    n_extra = (
        len(list(source_lp_result_0.instance.extra_galaxies))
        if source_lp_result_0.instance.extra_galaxies is not None
        else 0
    )
    n_scaling = (
        len(list(source_lp_result_0.instance.scaling_galaxies))
        if source_lp_result_0.instance.scaling_galaxies is not None
        else 0
    )

    tracer = (
        source_lp_result_0.max_log_likelihood_fit.tracer_linear_light_profiles_to_light_profiles
    )

    # Source MGE centred on primary lens bulge from stage 0
    source_bulge = al.model_util.mge_model_from(
        mask_radius=source_mge_radius,
        total_gaussians=30,
        centre_prior_is_uniform=False,
        centre=source_lp_result_0.instance.galaxies.lens_0.bulge.centre,
        centre_sigma=0.6,
    )

    # --- main lens full models (light fixed from stage 0, mass + shear free) ---
    # Only lens_0 carries the ExternalShear; one shear per group system.
    lens_full_models = []
    for i in range(n_main):
        lp0_lens = getattr(source_lp_result_0.instance.galaxies, f"lens_{i}")

        mass = af.Model(al.mp.Isothermal)
        mass.centre = lp0_lens.bulge.centre
        mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=5.0)

        lens_full_models.append(
            af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lp0_lens.bulge,
                disk=lp0_lens.disk,
                point=lp0_lens.point,
                mass=mass,
                shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
            )
        )

    # --- extra lens galaxy models (light fixed, mass bounded by luminosity) ---
    # Tracer order: [lens_0..lens_{n_main-1}, extra_0..extra_{n_extra-1}, scaling_0..]
    extra_lens_fixed_list = []
    for i in range(n_extra):
        lp0_extra = source_lp_result_0.instance.extra_galaxies[i]

        mass = af.Model(al.mp.Isothermal)
        mass.centre = lp0_extra.bulge.centre
        mass.ell_comps = lp0_extra.bulge.ell_comps

        luminosity_per_gaussian_list = [
            2 * np.pi * g.sigma**2 / g.axis_ratio() * g.intensity
            for g in tracer.galaxies[n_main + i].bulge.profile_list
        ]
        total_luminosity = np.sum(luminosity_per_gaussian_list) / pixel_scale**2
        mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0,
            upper_limit=min(5 * 0.5 * total_luminosity**0.6, 5.0),
        )

        extra_lens_fixed_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=lp0_extra.bulge, mass=mass)
        )

    extra_galaxies = af.Collection(extra_lens_fixed_list) if extra_lens_fixed_list else None

    # --- scaling lens galaxy models (light fixed, shared luminosity scaling relation) ---
    scaling_factor = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
    scaling_relation = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

    scaling_lens_fixed_list = []
    for i in range(n_scaling):
        lp0_scaling = source_lp_result_0.instance.scaling_galaxies[i]

        mass = af.Model(al.mp.Isothermal)
        mass.centre = lp0_scaling.bulge.centre
        mass.ell_comps = lp0_scaling.bulge.ell_comps

        luminosity_per_gaussian_list = [
            2 * np.pi * g.sigma**2 / g.axis_ratio() * g.intensity
            for g in tracer.galaxies[n_main + n_extra + i].bulge.profile_list
        ]
        total_luminosity = np.sum(luminosity_per_gaussian_list) / pixel_scale**2
        mass.einstein_radius = scaling_factor * total_luminosity**scaling_relation

        scaling_lens_fixed_list.append(
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=lp0_scaling.bulge, mass=mass
            )
        )

    scaling_galaxies = (
        af.Collection(scaling_lens_fixed_list) if scaling_lens_fixed_list else None
    )

    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_full_models)}
    lens_dict["source"] = af.Model(
        al.Galaxy, redshift=redshift_source, bulge=source_bulge
    )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
        scaling_galaxies=scaling_galaxies,
    )

    n_extra_model = len(extra_galaxies) if extra_galaxies is not None else 0
    n_scaling_model = len(scaling_galaxies) if scaling_galaxies is not None else 0
    n_live = 150 + 30 * n_main + 30 * n_extra_model + 30 * n_scaling_model

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
        n_like_max=200000,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def run_source_pix_1(
    dataset,
    mask,
    settings_search,
    source_lp_result_1,
    over_sample_size,
    pixel_scale,
    mask_radius,
    positions,
    n_batch=20,
):
    """
    __Model + Search + Analysis + Model-Fit (Search 3)__

    Search 3 of the SOURCE PIX PIPELINE fits a lens model where:

     - Every main lens galaxy has light fixed from source_lp[1] and a free Isothermal
       mass chained from source_lp[1] with unfix_mass_centre=True.
       ExternalShear is free on lens_0 only.
     - Extra and scaling galaxies retain the same models as source_lp[1] (fixed light,
       bounded/scaling mass).
     - The source galaxy has a free Delaunay pixelization with AdaptSplit regularization,
       adapted to the source signal estimated by source_lp[1].

    This search aims to produce an accurate pixelized source reconstruction with a
    well-constrained lens mass model.
    """
    hilbert_pixels = al.model_util.hilbert_pixels_from_pixel_scale(pixel_scale)
    edge_pixels_total = 30
    signal_to_noise_threshold = 3.0

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result_1
    )

    image_mesh = al.image_mesh.Hilbert(
        pixels=hilbert_pixels, weight_power=3.5, weight_floor=0.01
    )

    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=mask,
        adapt_data=galaxy_image_name_dict["('galaxies', 'source')"],
    )

    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=mask_radius + mask.pixel_scale / 2.0,
        n_points=edge_pixels_total,
    )

    adapt_images = al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid
        },
    )

    over_sample_size_pixelization = np.where(
        galaxy_image_name_dict["('galaxies', 'source')"] > signal_to_noise_threshold,
        4,
        2,
    )
    over_sample_size_pixelization = al.Array2D(
        values=over_sample_size_pixelization, mask=mask
    )

    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size,
        over_sample_size_pixelization=over_sample_size_pixelization,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result_1.positions_likelihood_from(
                factor=2.0, positions=positions, minimum_threshold=0.2
            )
        ],
    )

    n_lenses = sum(
        1 for k in vars(source_lp_result_1.instance.galaxies) if k.startswith("lens_")
    )

    lens_dict = {}
    for i in range(n_lenses):
        lp_lens_instance = getattr(source_lp_result_1.instance.galaxies, f"lens_{i}")
        lp_lens_model = getattr(source_lp_result_1.model.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=lp_lens_model.mass,
            mass_result=lp_lens_model.mass,
            unfix_mass_centre=True,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lp_lens_instance.redshift,
            bulge=lp_lens_instance.bulge,
            disk=lp_lens_instance.disk,
            point=lp_lens_instance.point,
            mass=mass,
            shear=lp_lens_model.shear,
        )

    lens_dict["source"] = af.Model(
        al.Galaxy,
        redshift=source_lp_result_1.instance.galaxies.source.redshift,
        pixelization=af.Model(
            al.Pixelization,
            mesh=al.mesh.Delaunay(pixels=hilbert_pixels, zeroed_pixels=edge_pixels_total),
            regularization=af.Model(al.reg.AdaptSplit),
        ),
    )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=source_lp_result_1.model.extra_galaxies,
        scaling_galaxies=source_lp_result_1.model.scaling_galaxies,
    )

    n_live = 150 + 50 * (n_lenses - 1)

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)
    return result, dataset, adapt_images


def run_source_pix_2(
    dataset,
    mask,
    settings_search,
    source_lp_result_1,
    source_pix_result_1,
    over_sample_size,
    pixel_scale,
    mask_radius,
    n_batch=20,
):
    """
    __Model + Search + Analysis + Model-Fit (Search 4)__

    Search 4 of the SOURCE PIX PIPELINE fits a lens model where:

     - Every main lens galaxy has light fixed from source_lp[1] and mass fixed from
       source_pix[1].
     - Extra and scaling galaxies are fully fixed from source_pix[1].
     - The source galaxy has a free Delaunay pixelization with AdaptSplit regularization,
       adapted to the source morphology from source_pix[1] via a Hilbert image mesh.

    This search aims to refine the pixelized source reconstruction using a more
    accurate adapt image derived from the first pixelized fit.
    """
    hilbert_pixels = al.model_util.hilbert_pixels_from_pixel_scale(pixel_scale)
    edge_pixels_total = 30
    signal_to_noise_threshold = 3.0
    signal_to_noise_threshold_image_mesh = 3.0

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_data_snr_max = galaxy_image_name_dict["('galaxies', 'source')"]
    adapt_data_snr_max[adapt_data_snr_max > signal_to_noise_threshold_image_mesh] = (
        signal_to_noise_threshold_image_mesh
    )

    image_mesh = al.image_mesh.Hilbert(
        pixels=hilbert_pixels, weight_power=3.5, weight_floor=0.01
    )

    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=mask, adapt_data=adapt_data_snr_max
    )

    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=mask_radius + mask.pixel_scale / 2.0,
        n_points=edge_pixels_total,
    )

    adapt_images = al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid
        },
    )

    over_sample_size_pixelization = np.where(
        galaxy_image_name_dict["('galaxies', 'source')"] > signal_to_noise_threshold,
        4,
        2,
    )
    over_sample_size_pixelization = al.Array2D(
        values=over_sample_size_pixelization, mask=mask
    )

    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size,
        over_sample_size_pixelization=over_sample_size_pixelization,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
    )

    n_lenses = sum(
        1 for k in vars(source_pix_result_1.instance.galaxies) if k.startswith("lens_")
    )

    lens_dict = {}
    for i in range(n_lenses):
        lp_lens_instance = getattr(source_lp_result_1.instance.galaxies, f"lens_{i}")
        pix1_lens_instance = getattr(source_pix_result_1.instance.galaxies, f"lens_{i}")

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lp_lens_instance.redshift,
            bulge=lp_lens_instance.bulge,
            disk=lp_lens_instance.disk,
            point=lp_lens_instance.point,
            mass=pix1_lens_instance.mass,
            shear=pix1_lens_instance.shear,
        )

    lens_dict["source"] = af.Model(
        al.Galaxy,
        redshift=source_lp_result_1.instance.galaxies.source.redshift,
        pixelization=af.Model(
            al.Pixelization,
            mesh=al.mesh.Delaunay(pixels=hilbert_pixels, zeroed_pixels=edge_pixels_total),
            regularization=af.Model(al.reg.AdaptSplit),
        ),
    )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=source_pix_result_1.instance.extra_galaxies,
        scaling_galaxies=source_pix_result_1.instance.scaling_galaxies,
    )

    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
        n_batch=n_batch,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)
    return result, dataset, adapt_images


def run_light_lp(
    dataset,
    settings_search,
    source_lp_result_0,
    source_pix_result_1,
    source_pix_result_2,
    adapt_images,
    mask_radius,
    redshift_lens,
    n_batch=20,
):
    """
    __Model + Search + Analysis + Model-Fit (Search 5)__

    Search 5 of the LIGHT LP PIPELINE fits a lens model where:

     - Every main lens galaxy has a free MGE bulge light profile; mass and shear are
       fixed from source_pix[1].
     - Extra galaxies have a free MGE bulge light profile and mass fixed from
       source_pix[1].
     - Scaling galaxies are fully fixed (light + mass) from source_pix[2].
     - The source is fixed (pixelized) from source_pix[2].

    This search aims to accurately model the lens galaxy light once the mass model
    and source are well constrained by the pixelized source pipeline.
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
    )

    n_lenses = sum(
        1 for k in vars(source_pix_result_1.instance.galaxies) if k.startswith("lens_")
    )
    n_extra = (
        len(list(source_pix_result_1.instance.extra_galaxies))
        if source_pix_result_1.instance.extra_galaxies is not None
        else 0
    )

    # --- main lens light models (MGE centred on stage-0 bulge centre) ---
    lens_bulge_list = []
    for i in range(n_lenses):
        lp0_lens = getattr(source_lp_result_0.instance.galaxies, f"lens_{i}")
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=30,
            gaussian_per_basis=2,
            centre_prior_is_uniform=False,
            centre=lp0_lens.bulge.centre,
            centre_sigma=0.1,
        )
        lens_bulge_list.append(bulge)

    # --- extra lens galaxy light models (free MGE, mass fixed from source_pix[1]) ---
    extra_lens_light_list = []
    for i in range(n_extra):
        pix1_extra = source_pix_result_1.instance.extra_galaxies[i]
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=10,
            centre_prior_is_uniform=False,
            centre=pix1_extra.mass.centre,
            centre_sigma=0.1,
        )
        extra_lens_light_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge, mass=pix1_extra.mass)
        )

    extra_galaxies = af.Collection(extra_lens_light_list) if extra_lens_light_list else None

    source = al.util.chaining.source_custom_model_from(
        result=source_pix_result_2, source_is_model=False
    )

    lens_dict = {}
    for i in range(n_lenses):
        lens_instance = getattr(source_pix_result_1.instance.galaxies, f"lens_{i}")

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_instance.redshift,
            bulge=lens_bulge_list[i],
            mass=lens_instance.mass,
            shear=lens_instance.shear,
        )

    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
        scaling_galaxies=source_pix_result_2.instance.scaling_galaxies,
    )

    n_live = 300 + 100 * (n_lenses - 1)

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def run_mass_total(
    dataset,
    settings_search,
    source_pix_result_1,
    source_pix_result_2,
    light_result,
    adapt_images,
    positions,
    pixel_scale,
    redshift_lens,
    n_batch=20,
):
    """
    __Model + Search + Analysis + Model-Fit (Search 6)__

    Search 6 of the MASS TOTAL PIPELINE fits a lens model where:

     - Every main lens galaxy has light fixed from light[1] and a free PowerLaw total
       mass distribution with priors chained from source_pix[1].
       ExternalShear is free on lens_0 only.
     - Extra galaxies have light fixed from light[1] and a free Isothermal mass bounded
       by the light[1] luminosity.
     - Scaling galaxies have light fixed from light[1] and mass set by a shared free
       luminosity scaling relation.
     - The source is fixed (pixelized) from source_pix[2].

    This search aims to accurately estimate the total mass distribution of every main
    lens galaxy, using the improved source and light models from preceding stages.
    """
    n_lenses = sum(
        1 for k in vars(light_result.instance.galaxies) if k.startswith("lens_")
    )
    n_extra = (
        len(list(light_result.instance.extra_galaxies))
        if light_result.instance.extra_galaxies is not None
        else 0
    )
    n_scaling = (
        len(list(light_result.instance.scaling_galaxies))
        if light_result.instance.scaling_galaxies is not None
        else 0
    )

    # --- extra galaxies: fixed light, free mass ---
    extra_lens_mass_free_list = []
    for i in range(n_extra):
        light_extra = light_result.instance.extra_galaxies[i]

        mass = af.Model(al.mp.Isothermal)
        mass.centre = light_extra.bulge.centre
        mass.ell_comps = light_extra.bulge.ell_comps

        luminosity_per_gaussian_list = [
            2 * np.pi * g.sigma**2 / g.axis_ratio() * g.intensity
            for g in light_extra.bulge.profile_list
        ]
        total_luminosity = np.sum(luminosity_per_gaussian_list) / pixel_scale**2
        mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0,
            upper_limit=min(5 * 0.5 * total_luminosity**0.6, 5.0),
        )

        extra_lens_mass_free_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=light_extra.bulge, mass=mass)
        )

    extra_galaxies = (
        af.Collection(extra_lens_mass_free_list) if extra_lens_mass_free_list else None
    )

    # --- scaling galaxies: fixed light, free shared scaling relation ---
    scaling_factor = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
    scaling_relation = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

    scaling_mass_free_list = []
    for i in range(n_scaling):
        light_scaling = light_result.instance.scaling_galaxies[i]

        mass = af.Model(al.mp.Isothermal)
        mass.centre = light_scaling.bulge.centre
        mass.ell_comps = light_scaling.bulge.ell_comps

        luminosity_per_gaussian_list = [
            2 * np.pi * g.sigma**2 / g.axis_ratio() * g.intensity
            for g in light_scaling.bulge.profile_list
        ]
        total_luminosity = np.sum(luminosity_per_gaussian_list) / pixel_scale**2
        mass.einstein_radius = scaling_factor * total_luminosity**scaling_relation

        scaling_mass_free_list.append(
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=light_scaling.bulge, mass=mass
            )
        )

    scaling_galaxies = (
        af.Collection(scaling_mass_free_list) if scaling_mass_free_list else None
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            light_result.positions_likelihood_from(
                factor=3.0, positions=positions, minimum_threshold=0.2
            )
        ],
    )

    source = al.util.chaining.source_from(result=source_pix_result_2)

    lens_dict = {}
    for i in range(n_lenses):
        lens_model = getattr(source_pix_result_1.model.galaxies, f"lens_{i}")
        light_lens_instance = getattr(light_result.instance.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=af.Model(al.mp.PowerLaw),
            mass_result=lens_model.mass,
            unfix_mass_centre=True,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_model.redshift,
            bulge=light_lens_instance.bulge,
            disk=light_lens_instance.disk,
            point=light_lens_instance.point,
            mass=mass,
            shear=lens_model.shear,
        )

    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
        scaling_galaxies=scaling_galaxies,
    )

    n_live = 200 + 100 * (n_lenses - 1)

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def fit(dataset_name, sample_name):
    """
    __Paths__
    """
    project_root = Path(__file__).parent.parent

    config_path = project_root / "config"
    dataset_path = project_root / "dataset"
    output_path = project_root / "output"

    if sample_name is not None:
        dataset_path = dataset_path / sample_name
    if dataset_name is not None:
        dataset_path = dataset_path / dataset_name

    """
    __Info__
    """
    with open(dataset_path / "info.json") as json_file:
        info = json.load(json_file)

    pixel_scale = info.get("pixel_scale", 0.1)
    mask_radius = info.get("mask_radius", 6.0)
    mask_centre = info.get("mask_centre", (0.0, 0.0))
    redshift_lens = info.get("redshift_lens", 0.5)
    redshift_source = info.get("redshift_source", 1.0)
    source_mge_radius = info.get("source_mge_radius", 1.0)
    n_batch = info.get("n_batch", 20)

    """
    __Configs__
    """
    from autoconf import conf

    conf.instance.push(new_path=config_path, output_path=output_path)

    """
    __Dataset__
    """
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=pixel_scale,
    )

    try:
        mask_extra_galaxies = al.Mask2D.from_fits(
            file_path=dataset_path / "mask_extra_galaxies.fits",
            pixel_scales=pixel_scale,
            invert=True,
        )
        dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
    except FileNotFoundError:
        pass

    """
    __Galaxy Centres__

    main_lens_centres.json        — required; determines the number of main lenses.
    extra_galaxies_centres.json  — optional; empty list if absent.
    scaling_galaxies_centres.json — optional; empty list if absent.

    All three files contain a list of [y, x] arcsecond coordinates.
    """
    main_lens_centres = _load_centres(dataset_path / "main_lens_centres.json")
    extra_lens_centres = _load_centres(dataset_path / "extra_galaxies_centres.json")
    scaling_lens_centres = _load_centres(dataset_path / "scaling_galaxies_centres.json")

    all_galaxy_centres = al.Grid2DIrregular(
        main_lens_centres.in_list
        + extra_lens_centres.in_list
        + scaling_lens_centres.in_list
    )

    positions = al.Grid2DIrregular(
        al.from_json(file_path=dataset_path / "positions.json")
    )

    """
    __Mask__
    """
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
        centre=mask_centre,
    )

    dataset = dataset.apply_mask(mask=mask)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.1, 0.3],
        centre_list=list(all_galaxy_centres),
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    """
    __Settings AutoFit__
    """
    settings_search = af.SettingsSearch(
        path_prefix=sample_name,
        unique_tag=dataset_name,
        info=info,
        session=None,
    )

    """
    __Pipeline__
    """
    source_lp_result_0 = run_source_lp_0(
        dataset=dataset,
        settings_search=settings_search,
        main_lens_centres=main_lens_centres,
        extra_lens_centres=extra_lens_centres,
        scaling_lens_centres=scaling_lens_centres,
        mask_radius=mask_radius,
        redshift_lens=redshift_lens,
    )

    source_lp_result_1 = run_source_lp_1(
        dataset=dataset,
        settings_search=settings_search,
        source_lp_result_0=source_lp_result_0,
        positions=positions,
        pixel_scale=pixel_scale,
        redshift_lens=redshift_lens,
        redshift_source=redshift_source,
        source_mge_radius=source_mge_radius,
    )

    source_pix_result_1, dataset, adapt_images = run_source_pix_1(
        dataset=dataset,
        mask=mask,
        settings_search=settings_search,
        source_lp_result_1=source_lp_result_1,
        over_sample_size=over_sample_size,
        pixel_scale=pixel_scale,
        mask_radius=mask_radius,
        positions=positions,
        n_batch=n_batch,
    )

    source_pix_result_2, dataset, adapt_images = run_source_pix_2(
        dataset=dataset,
        mask=mask,
        settings_search=settings_search,
        source_lp_result_1=source_lp_result_1,
        source_pix_result_1=source_pix_result_1,
        over_sample_size=over_sample_size,
        pixel_scale=pixel_scale,
        mask_radius=mask_radius,
        n_batch=n_batch,
    )

    light_result = run_light_lp(
        dataset=dataset,
        settings_search=settings_search,
        source_lp_result_0=source_lp_result_0,
        source_pix_result_1=source_pix_result_1,
        source_pix_result_2=source_pix_result_2,
        adapt_images=adapt_images,
        mask_radius=mask_radius,
        redshift_lens=redshift_lens,
        n_batch=n_batch,
    )

    mass_result = run_mass_total(
        dataset=dataset,
        settings_search=settings_search,
        source_pix_result_1=source_pix_result_1,
        source_pix_result_2=source_pix_result_2,
        light_result=light_result,
        adapt_images=adapt_images,
        positions=positions,
        pixel_scale=pixel_scale,
        redshift_lens=redshift_lens,
        n_batch=n_batch,
    )

    return (
        source_lp_result_0,
        source_lp_result_1,
        source_pix_result_1,
        source_pix_result_2,
        light_result,
        mass_result,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PyAutoLens SLAM Group Pipeline — Multiple Main Lenses"
    )

    parser.add_argument(
        "--sample",
        metavar="name",
        required=False,
        default=None,
        help="Name of the sample subdirectory inside dataset/ (e.g. euclid_groups).",
    )

    parser.add_argument(
        "--dataset",
        metavar="name",
        required=False,
        default=None,
        help="Name of the dataset subdirectory inside dataset/[sample]/ (e.g. group0001).",
    )

    args = parser.parse_args()

    fit(dataset_name=args.dataset, sample_name=args.sample)


"""
Finish.
"""
