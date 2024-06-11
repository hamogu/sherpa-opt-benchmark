# Licensed under GLPv3 - see LICENSE
"""Provide datasets for the benchmark fitting to run on.

This module provides functions that return datasets for the benchmark
fitting to run on. Each function returns a tuple with the following
elements:

    - pha dataset
    - full model that should be it (incl ARF and RMF)
    - true values of the parameters
    - starting values for the fit

Function that start with `data_` are automatically discovered and used
in the tests.

Follow the structure in this file to add more datasets.
"""

from functools import lru_cache

import numpy as np
from sherpa.astro.io import read_pha
from sherpa.astro.fake import fake_pha
from sherpa.astro import xspec
from sherpa.astro.instrument import Response1D


rng = np.random.default_rng()  # fixed seed for reproducibility?


@lru_cache
def make_acis_powerlaw():
    """Create a dataset with an absorbed power law model

    We use a real Chandra/ACIS CCD ARF and RMF and simulate an
    absorbed power law model. The number of counts is low, so we group
    and we will often find significant deviations between the input
    numbers and the output of the fit just from statistical fluctuations.

    The return value from this function is cached. That way, the same
    spectrum is given to all optimizers, despite using a random generator
    to create the data.
    (It's also faster, but that does not matter much.)
    """
    pha = read_pha(
        "../curated-test-data/Chandra/ACIS/acisf04487_001N023_r0009_pha3.fits.gz"
    )
    gal = xspec.XSphabs()
    clus = xspec.XSapec()
    model = gal * clus
    start_pars = model.thawedpars

    gal.nh = 0.12
    clus.kt = 4.5
    clus.Abundanc = 0.3
    clus.redshift = 0.23
    clus.norm = 1.2e-3
    resp = Response1D(pha)
    full_model = resp(model)

    fake_pha(pha, full_model, rng=rng)
    pha.group_counts(5)
    pha.ignore(None, 0.5)
    pha.ignore(7, None)
    return pha, full_model, full_model.thawedpars, start_pars


def data_acis_abspowerlaw():
    """Create a dataset with an absorbed power law model

    We use a real Chandra/ACIS CCD ARF and RMF and simulate an
    absorbed power law model. The number of counts is low, so we group
    and we will often find significant deviations between the input
    numbers and the output of the fit just from statistical fluctuations.
    """
    return make_acis_powerlaw()


def data_acis_abspowerlaw_close():
    """Same setup as above but start the fit close to the final values

    Here, we start the fit with the same model that was used to generate the data.
    Due to statistical fluctuations, the fit will be a little bit off from those numbers,
    but it should be close, so we get a  case where very few optimization steps are needed.
    """
    pha, full_model, truepars, _ = make_acis_powerlaw()
    return pha, full_model, truepars, truepars


def data_MOS_DGTau():
    pha = read_pha("curated-test-data/XMM-Newton/EPIC-MOS1/MOS1_spectrum_grp.fits")
    abs1 = xspec.XSphabs(name="jetabs")
    abs2 = xspec.XSphabs(name="jetemiss")
    vapec1 = xspec.XSvapec(name="starabs")
    vapec2 = xspec.XSvapec(name="staremiss")
    model = abs1 * vapec1 + abs2 * vapec2

    # set some parameters to non-default values
    abs1.nH = 0.1
    abs2.nH = 1.0
    # This is a higher flux tan DG Tau has, but it's also a
    # short exposure time and we need enough counts to fit.
    vapec1.norm = 5e-4
    vapec2.norm = 5e-4
    for elem in ["Fe", "Ne"]:
        setattr(vapec2, elem, getattr(vapec1, elem))
        getattr(vapec1, elem).frozen = False

    # Start the fit so the first component is the one with the
    # lower temperature
    vapec1.kT = 0.5
    vapec2.kT = 5.0
    start_pars = model.thawedpars

    # But change the true values
    # so the the fit still has to change something
    vapec1.kT = 0.3
    vapec2.kT = 2.5

    # Just do the most important elements here.
    # The point is that they are not all at 1.
    vapec1.Ne = 2.5
    vapec2.Fe = 0.3
    resp = Response1D(pha)
    full_model = resp(model)

    fake_pha(pha, full_model, rng=rng)
    pha.group_counts(5)
    pha.ignore(None, 0.5)
    pha.ignore(7, None)
    return pha, full_model, full_model.thawedpars, start_pars