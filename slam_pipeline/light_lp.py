import autofit as af
import autogalaxy as ag
import autolens as al

from typing import List, Union, Optional


def run(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    pixel_scales: Optional[float] = None,
    lens_bulge: Optional[af.Model] = af.Model(al.lp.Sersic),
    lens_disk: Optional[af.Model] = None,
    lens_point: Optional[af.Model] = None,
    extra_galaxies: Optional[af.Collection] = None,
    dataset_model: Optional[af.Model] = None,
    n_batch: int = 20,
) -> af.Result:
    """
    The SlaM LIGHT LP PIPELINE, which fits a complex model for a lens galaxy's light with the mass and source models
    fixed.

    Parameters
    ----------
    settings_search
        A collection of settings that control the behaviour of PyAutoFit thoughout the pipeline (e.g. paths, database,
        parallelization, etc.).
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    source_result_for_lens
        The result of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline, used
        for initializing model components associated with the lens galaxy.
    source_result_for_source
        The result of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline, used
        for initializing model components associated with the source galaxy.
    pixel_scales
        The pixel scale of the dataset in arcseconds per pixel.  When provided and ``lens_point`` is ``None``,
        a compact MGE Basis model is automatically constructed via
        ``ag.model_util.mge_point_model_from(pixel_scales=pixel_scales)`` and used as ``lens_point``.
    lens_bulge
        The model used to represent the light distribution of the lens galaxy's bulge (set to
        None to omit a bulge).
    lens_disk
        The model used to represent the light distribution of the lens galaxy's disk (set to
        None to omit a disk).
    lens_point
        The model used to represent the light distribution of the lens galaxy's point-source(s)
        emission (e.g. a nuclear star burst region) or compact central structures (e.g. an unresolved bulge).
        When ``None`` and ``pixel_scales`` is provided, defaults to an MGE Basis of linear Gaussians whose
        sigma values span 0.01 arcseconds to ``2 * pixel_scales`` (see ``ag.model_util.mge_point_model_from``).
    extra_galaxies
        Additional extra galaxies containing light and mass profiles, which model nearby line of sight galaxies.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    """

    if lens_point is None and pixel_scales is not None:
        lens_point = ag.model_util.mge_point_model_from(pixel_scales=pixel_scales)

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the LIGHT LP PIPELINE fits a lens model where:

     - The lens galaxy light is modeled using a light profiles [no prior initialization].
     - The lens galaxy mass is modeled using SOURCE PIPELINE's mass distribution [Parameters fixed from SOURCE PIPELINE].
     - The source galaxy's light is modeled using SOURCE PIPELINE's model [Parameters fixed from SOURCE PIPELINE].

    This search aims to produce an accurate model of the lens galaxy's light, which may not have been possible in the
    SOURCE PIPELINE as the mass and source models were not properly initialized.
    """

    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=lens_bulge,
                disk=lens_disk,
                point=lens_point,
                mass=source_result_for_lens.instance.galaxies.lens.mass,
                shear=source_result_for_lens.instance.galaxies.lens.shear,
            ),
            source=source,
        ),
        extra_galaxies=extra_galaxies,
        dataset_model=dataset_model,
    )

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=300,
        n_batch=n_batch,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result


def run_group(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    lens_bulge_list: Optional[List] = None,
    lens_disk_list: Optional[List] = None,
    lens_point_list: Optional[List] = None,
    extra_galaxies: Optional[af.Collection] = None,
    scaling_galaxies: Optional[af.Collection] = None,
    dataset_model: Optional[af.Model] = None,
    n_batch: int = 20,
) -> af.Result:
    """
    The SLaM LIGHT LP PIPELINE for group-scale lenses with multiple main lens galaxies.

    Group-scale counterpart to ``run``.  Fits free light profiles for every main lens
    galaxy while fixing mass (from ``source_result_for_lens``) and source (from
    ``source_result_for_source``).

    Parameters
    ----------
    settings_search
        AutoFit settings controlling output paths and search configuration.
    analysis
        The analysis class containing the log-likelihood function.
    source_result_for_lens
        Result whose ``instance.galaxies.lens_i`` provides the fixed mass and shear
        for each main lens.  Typically ``source_pix_result_2``.
    source_result_for_source
        Result whose source galaxy is fixed.  Typically ``source_pix_result_2``.
    lens_bulge_list
        One bulge model per main lens.  ``None`` entries omit the bulge for that lens.
        If the list itself is ``None``, defaults to ``af.Model(al.lp.Sersic)`` for
        every lens.
    lens_disk_list
        One disk model per main lens.  ``None`` entries or a ``None`` list omit the
        disk.
    lens_point_list
        One point model per main lens.  ``None`` entries or a ``None`` list omit the
        point component.
    extra_galaxies
        Extra galaxies with light and mass fixed from the preceding source result.
    scaling_galaxies
        Scaling galaxies with light and mass fixed from the preceding source result.
    dataset_model
        Optional dataset-level model components (e.g. background sky, astrometric
        offsets for multi-band fitting).
    n_batch
        Nautilus batch size.
    """
    n_lenses = sum(
        1
        for k in vars(source_result_for_lens.instance.galaxies)
        if k.startswith("lens_")
    )

    if lens_bulge_list is None:
        lens_bulge_list = [af.Model(al.lp.Sersic) for _ in range(n_lenses)]
    if lens_disk_list is None:
        lens_disk_list = [None] * n_lenses
    if lens_point_list is None:
        lens_point_list = [None] * n_lenses

    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    lens_dict = {}
    for i in range(n_lenses):
        lens_instance = getattr(source_result_for_lens.instance.galaxies, f"lens_{i}")

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_instance.redshift,
            bulge=lens_bulge_list[i],
            disk=lens_disk_list[i],
            point=lens_point_list[i],
            mass=lens_instance.mass,
            shear=lens_instance.shear,
        )

    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
        scaling_galaxies=scaling_galaxies,
        dataset_model=dataset_model,
    )

    n_live = 300 + 100 * (n_lenses - 1)

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)
