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

Pipeline stages implemented here
---------------------------------
source_lp[0]  run_0__group  — light only, no mass, no source
source_lp[1]  run_group     — full LP fit, mass + source introduced

Subsequent stages (source_pix, light_lp, mass_total) will be added once the
group variants of those pipeline functions are implemented.
"""

import numpy as np
import sys
import json
from pathlib import Path


def _load_centres(path):
    """Load a centres JSON file, returning an empty list if the file is absent."""
    try:
        import autolens as al

        return al.Grid2DIrregular(al.from_json(file_path=path))
    except FileNotFoundError:
        import autolens as al

        return al.Grid2DIrregular([])


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
    # Sigma range for the source MGE: should cover the source arc but not the lens
    # light.  A reasonable starting point is ~half the Einstein radius; override
    # per-dataset in info.json as needed.
    source_mge_radius = info.get("source_mge_radius", 1.0)

    """
    __Configs__
    """
    from autoconf import conf

    conf.instance.push(new_path=config_path, output_path=output_path)

    """
    __AutoLens + Data__
    """
    import autofit as af
    import autolens as al


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
    __SOURCE LP PIPELINE — stage 0: light only__

    Fits MGE light profiles for every main lens galaxy simultaneously with no
    mass model and no source. Extra lens galaxies are also fit with light-only
    MGE profiles.

    The result initializes:
      - bulge centres for each main lens (used as mass_centre in stage 1)
      - extra lens galaxy luminosities (used to bound mass Einstein radii in stage 1)
    """
    """
    __Analysis__

    No adapt images or positions likelihood; light-only, no lensing geometry to constrain.
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
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=bulge, disk=None, point=None
            )
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

    extra_galaxies_free = (
        af.Collection(extra_lens_light_list) if extra_lens_light_list else None
    )

    # --- scaling galaxy light models (same MGE profile; kept separate so the two
    #     populations remain distinguishable when building stage-1 mass models) ---
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

    scaling_galaxies_free = (
        af.Collection(scaling_light_list) if scaling_light_list else None
    )

    lens_galaxy_models = lens_light_models
    extra_galaxies = extra_galaxies_free
    scaling_galaxies = scaling_galaxies_free

    """
    __Model__

    source_lp[0] fits a model where:

     - Every main lens galaxy has a free MGE light profile (no mass, no source).
     - Every extra galaxy has a free MGE light profile.
     - Every scaling galaxy has a free MGE light profile.
    """
    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_galaxy_models)}

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
        scaling_galaxies=scaling_galaxies,
    )

    n_extra = len(extra_galaxies) if extra_galaxies is not None else 0
    n_scaling = len(scaling_galaxies) if scaling_galaxies is not None else 0
    n_live = 100 + 30 * len(lens_galaxy_models) + 30 * n_extra + 30 * n_scaling

    """
    __Search__

    n_live scales with the total number of galaxies across all three populations.
    """
    search = af.Nautilus(
        name="source_lp[0]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=50,
        n_like_max=1000000,
    )

    source_lp_result_0 = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    """
    __SOURCE LP PIPELINE — stage 1: full LP fit__

    Introduces mass and source. For each main lens:
      - bulge is fixed to the stage-0 instance (light already well-characterized)
      - mass centre is pinned to the stage-0 bulge centre
      - mass Einstein radius has a broad uniform prior
      - ExternalShear is free

    Extra lens galaxies have light fixed from stage 0; mass Einstein radii are
    bounded by the stage-0 luminosity (same luminosity-scaling bound as group.py).

    Scaling lens galaxies have mass modeled via a shared luminosity scaling relation.
    """
    """
    __Analysis__

    Positions likelihood guards against catastrophic lens configurations from the start.
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[al.PositionsLH(positions=positions, threshold=0.3)],
    )

    source_bulge = al.model_util.mge_model_from(
        mask_radius=source_mge_radius,
        total_gaussians=30,
        centre_prior_is_uniform=False,
        centre=(main_lens_centres[0][0], main_lens_centres[0][1]),
        centre_sigma=0.6,
    )

    # --- main lens full models (light fixed from stage 0, mass + shear free) ---
    # Only lens_0 carries the ExternalShear; there is one shear per group system,
    # not one per galaxy.  lens_0 should be the largest/primary galaxy.
    lens_full_models = []
    for i in range(len(main_lens_centres)):
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

    # Tracer galaxy order mirrors the model collection order:
    #   indices 0 .. n_main-1                   → main lenses
    #   indices n_main .. n_main+n_extra-1       → extra_galaxies
    #   indices n_main+n_extra .. end            → scaling_galaxies
    n_main = len(main_lens_centres)
    n_extra = len(extra_lens_centres)

    tracer = (
        source_lp_result_0.max_log_likelihood_fit.tracer_linear_light_profiles_to_light_profiles
    )

    # --- extra lens galaxy models (light fixed, mass bounded by luminosity) ---
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
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=lp0_extra.bulge, mass=mass
            )
        )

    extra_galaxies_fixed = (
        af.Collection(extra_lens_fixed_list) if extra_lens_fixed_list else None
    )

    # --- scaling lens galaxy models (light fixed, shared luminosity scaling relation) ---
    # scaling_galaxies were merged into extra_galaxies inside run_0__group so that
    # autolens includes them in the tracer.  Retrieve them by offset here.
    scaling_factor = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
    scaling_relation = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

    scaling_lens_fixed_list = []
    for i in range(len(scaling_lens_centres)):
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

    lens_galaxy_models = lens_full_models
    extra_galaxies = extra_galaxies_fixed
    scaling_galaxies = scaling_galaxies

    """
    __Model__

    source_lp[1] fits a model where:

     - Every main lens galaxy has light fixed from [0], a free Isothermal mass, and
       ExternalShear on lens_0 only.
     - Every extra galaxy has light fixed from [0] and a free Isothermal mass bounded
       by luminosity.
     - Every scaling galaxy has light fixed from [0] and mass set by a shared free
       scaling relation.
     - The source galaxy is a free MGE light profile.
    """
    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_galaxy_models)}
    lens_dict["source"] = af.Model(
        al.Galaxy,
        redshift=redshift_source,
        bulge=source_bulge,
    )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
        scaling_galaxies=scaling_galaxies,
    )

    n_extra = len(extra_galaxies) if extra_galaxies is not None else 0
    n_scaling = len(scaling_galaxies) if scaling_galaxies is not None else 0
    n_live = 150 + 30 * len(lens_galaxy_models) + 30 * n_extra + 30 * n_scaling

    """
    __Search__

    n_live scales with number of lenses, extra, and scaling galaxies.
    """
    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=50,
        n_like_max=200000,
    )

    source_lp_result_1 = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    """
    __SOURCE PIX PIPELINE 1__

    Introduces a pixelized source reconstruction.  The adapt image is built from
    the source signal estimated by source_lp[1].

    All main lens light profiles are fixed from source_lp[1].  Mass models are
    free, chained from source_lp[1] with the centre unfixed.  The single
    ExternalShear (on lens_0) is also kept free.

    Extra and scaling galaxies are passed in with the same models used in
    source_lp[1] (fixed light, bounded/scaling mass).
    """
    hilbert_pixels = al.model_util.hilbert_pixels_from_pixel_scale(pixel_scale)

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result_1
    )

    image_mesh = al.image_mesh.Hilbert(
        pixels=hilbert_pixels, weight_power=3.5, weight_floor=0.01
    )

    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask,
        adapt_data=galaxy_image_name_dict["('galaxies', 'source')"],
    )

    edge_pixels_total = 30

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

    signal_to_noise_threshold = 3.0
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

    """
    __Analysis__

    Adapt images from source_lp[1]; positions likelihood tightened from source_lp[1] at factor=2.
    """
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

    """
    __Model__

    source_pix[1] fits a model where:

     - Every main lens galaxy has light fixed from source_lp[1] and a free Isothermal
       mass chained with unfix_mass_centre=True; ExternalShear free on lens_0 only.
     - Extra and scaling galaxies are fixed (light + mass) from source_lp[1].
     - The source galaxy has a free Delaunay pixelization + AdaptSplit regularization.
    """
    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies_fixed,
        scaling_galaxies=scaling_galaxies,
    )

    n_live = 150 + 50 * (n_lenses - 1)

    """
    __Search__

    n_live scales with the number of free mass models (one per main lens).
    """
    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=info.get("n_batch", 20),
    )

    source_pix_result_1 = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    """
    __SOURCE PIX PIPELINE 2__

    Refines the pixelization using a Hilbert image mesh adapted to the source
    morphology from source_pix[1].  All lens components (light and mass) and
    extra_galaxies are fully fixed.
    """
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    image_mesh = al.image_mesh.Hilbert(
        pixels=hilbert_pixels, weight_power=3.5, weight_floor=0.01
    )

    signal_to_noise_threshold_image_mesh = 3.0
    adapt_data_snr_max = galaxy_image_name_dict["('galaxies', 'source')"]
    adapt_data_snr_max[adapt_data_snr_max > signal_to_noise_threshold_image_mesh] = (
        signal_to_noise_threshold_image_mesh
    )

    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask, adapt_data=adapt_data_snr_max
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

    """
    __Analysis__

    Hilbert adapt images from source_pix[1]; no positions likelihood — everything is fixed.
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
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

    """
    __Model__

    source_pix[2] fits a model where:

     - Every main lens galaxy has light fixed from source_lp[1] and mass fixed from
       source_pix[1].
     - Extra and scaling galaxies are fully fixed from source_pix[1].
     - The source galaxy has a free Delaunay pixelization + AdaptSplit regularization.
    """
    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=source_pix_result_1.instance.extra_galaxies,
        scaling_galaxies=source_pix_result_1.instance.scaling_galaxies,
    )

    """
    __Search__

    Fixed n_live=75; only the pixelization mesh and regularization are free.
    """
    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
        n_batch=info.get("n_batch", 20),
    )

    source_pix_result_2 = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    """
    __LIGHT LP PIPELINE__

    Fits free MGE light profiles for every main lens galaxy simultaneously while
    holding the mass models (from source_pix[2]) and source (pixelized, fixed) in
    place.  Extra and scaling galaxies are fully fixed (light + mass) from the
    source_pix[2] result.

    The same MGE setup used in source_lp[0] is used here — one per main lens,
    centred on the stage-0 bulge centre.  This gives a flexible, high-dynamic-range
    light model that can properly separate lens from source once the mass model is
    well-constrained.
    """
    """
    __Analysis__

    Hilbert adapt images from source_pix[2]; no positions likelihood.
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
    )

    lens_bulge_list = []
    for i in range(len(main_lens_centres)):
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

    # --- extra lens galaxy light models ---
    extra_lens_light_list = []
    for i, centre in enumerate(extra_lens_centres):
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=10,
            centre_prior_is_uniform=False,
            centre=(centre[0], centre[1]),
            centre_sigma=0.1,
        )
        mass = source_pix_result_1.instance.extra_galaxies[i].mass
        extra_lens_light_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge, mass=mass)
        )

    extra_galaxies_free = (
        af.Collection(extra_lens_light_list) if extra_lens_light_list else None
    )

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

    """
    __Model__

    light[1] fits a model where:

     - Every main lens galaxy has a free MGE bulge; mass and shear fixed from
       source_pix[1].
     - Extra galaxies have a free MGE bulge and mass fixed from source_pix[1].
     - Scaling galaxies are fully fixed from source_pix[2].
     - The source is fixed (pixelized) from source_pix[2].
    """
    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies_free,
        scaling_galaxies=source_pix_result_2.instance.scaling_galaxies,
    )

    n_live = 300 + 100 * (n_lenses - 1)

    """
    __Search__

    n_live scales with number of main lenses; light is the only free component.
    """
    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=info.get("n_batch", 20),
    )

    light_result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    """
    __MASS TOTAL PIPELINE__

    Fits a PowerLaw total mass distribution for every main lens simultaneously.
    Mass priors are chained from source_pix[1].  Light is fixed from light[1].
    Source is fixed from source_pix[2].

    Extra galaxies: light fixed from light[1], mass free (Isothermal, centre
    fixed to bulge centre, einstein_radius bounded by luminosity).

    Scaling galaxies: light fixed from light[1], mass driven by a shared free
    scaling relation (scaling_factor and scaling_relation both free).
    """
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
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=light_extra.bulge, mass=mass
            )
        )

    extra_galaxies_mass_free = (
        af.Collection(extra_lens_mass_free_list) if extra_lens_mass_free_list else None
    )

    # --- scaling galaxies: fixed light, free shared scaling relation ---
    scaling_factor = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
    scaling_relation = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

    scaling_mass_free_list = []
    for i in range(len(scaling_lens_centres)):
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
                al.Galaxy,
                redshift=redshift_lens,
                bulge=light_scaling.bulge,
                mass=mass,
            )
        )

    scaling_galaxies_mass_free = (
        af.Collection(scaling_mass_free_list) if scaling_mass_free_list else None
    )

    """
    __Analysis__

    Adapt images from source_pix[2]; positions likelihood from light[1] at factor=3.
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            light_result.positions_likelihood_from(
                factor=3.0, positions=positions, minimum_threshold=0.2
            )
        ],
    )

    """
    __Model__

    mass_total[1] fits a model where:

     - Every main lens galaxy has light fixed from light[1] and a free PowerLaw mass
       with priors chained from source_pix[1]; ExternalShear free on lens_0 only.
     - Extra galaxies have light fixed from light[1] and a free Isothermal mass
       bounded by luminosity.
     - Scaling galaxies have light fixed from light[1] and mass set by a shared free
       scaling relation.
     - The source is fixed (pixelized) from source_pix[2].
    """
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
        extra_galaxies=extra_galaxies_mass_free,
        scaling_galaxies=scaling_galaxies_mass_free,
    )

    n_live = 200 + 100 * (n_lenses - 1)

    """
    __Search__

    n_live scales with number of main lenses; PowerLaw replaces Isothermal from source pipeline.
    """
    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=info.get("n_batch", 20),
    )

    mass_result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

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
