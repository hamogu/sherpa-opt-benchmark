# Licensed under GLPv3 - see LICENSE
"""Code to run benchmark

The main function in this module is `test_optimizers`.
It takes datasets and optimizers and runs the optimzation, then
produces a number of outputs. Some of those are simply collected in
a `results_bag`, others are written out into the `results` directory on
disk and include a table of model parameters and a plot of the fit.

The `test_synthesis` function is a simple function that takes the
results from the `results_bag` and produces a summary table that is
printed to the console and saves in the results directory.

The `optimizer` and `dataset` fixtures are parametrized with the
cases from the `optimizers` and `datasets_pha` modules, respectively.
"""

from pathlib import Path
import re

from sherpa.fit import Fit
from sherpa.stats import Cash
from sherpa.astro.plot import DataPHAPlot, ModelHistogram
from matplotlib import pyplot as plt
import pandas as pd

from pytest_cases import fixture, parametrize_with_cases


@fixture(scope="session")
@parametrize_with_cases("opt", cases=".optimizers", prefix="opt_")
def optimizer(opt):
    yield opt


@fixture(scope="session")
@parametrize_with_cases("data", cases=".datasets_pha", prefix="data_")
def dataset(data):
    # (optional setup code here)
    yield data
    # (optional teardown code here)


def test_optimizers(optimizer, dataset, results_bag, request):
    # Create output dir is needed
    output_dir = (
        Path(__file__).parent.parent
        / "results"
        / request.node.name.replace("[", "_").replace("]", "")
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    pha, full_model, true_vals, start_vals = dataset

    # Need to reset to starting values, because we reuse the same model
    for p, v in zip(full_model.get_thawed_pars(), start_vals):
        p.val = v

    # Fit the model
    f = Fit(pha, full_model, Cash(), optimizer)
    fit_res = f.fit(outfile=output_dir / "fitsteps.dat", clobber=True)
    fitted_pars = full_model.thawedpars

    # write out model parameters
    model_df = pd.DataFrame(
        {
            "names": [p.fullname for p in full_model.get_thawed_pars()],
            "true": true_vals,
            "start": start_vals,
            "fit": fitted_pars,
        }
    )
    model_df.to_csv(output_dir / "model_pars.csv")

    # First put stuff in the results_bag
    # Then assert that the fit was successful
    # because when an assert fails, the test stops
    results_bag.optimizer = optimizer.name
    results_bag.statval = fit_res.statval
    results_bag.nfev = fit_res.nfev
    for k, v in zip(fit_res.parnames, fit_res.parvals):
        results_bag[k] = v

    # Plot the fit results for visual inspection
    dplot = DataPHAPlot()
    mplot = ModelHistogram()
    dplot.prepare(pha)
    dplot.plot(xlog=True)
    mplot = ModelHistogram()
    mplot.prepare(pha, full_model)
    mplot.overplot()
    for p, v in zip(full_model.get_thawed_pars(), true_vals):
        p.val = v
    mplot.prepare(pha, full_model)
    mplot.overplot()
    ax = plt.gca()
    ax.set_title(request.node.name)
    fig = plt.gcf()
    fig.savefig(output_dir / "fitplot.png")

    assert fit_res.succeeded
    # assert fitted_pars == pytest.approx(true_vals, rel=0.1)


dataname = re.compile("CombinedFixtureParamValue\(<\['data'\]: (?P<dataname>\w+)>\)")


def test_synthesis(module_results_df):
    # `module_results_df` already contains everything at this point.

    # Extract the dataset name from the dataset_param
    module_results_df["dataset"] = [
        dataname.search(str(d)).groupdict()["dataname"]
        for d in module_results_df["dataset_param"]
    ]
    # module_results_df["dataset_param"] = module_results_df["dataset_param"].map(str)
    # module_results_df = module_results_df[
    #    ["dataset", "optimizer", "apec.norm", "status", "duration_ms", "statval"]
    # ]
    module_results_df = module_results_df.drop(
        columns=["pytest_obj", "dataset_param", "optimizer_param"]
    )
    # print the synthesis dataframe

    print("\n Collected benchmark results:\n")
    print(module_results_df)
    output_dir = Path(__file__).parent.parent / "results"
    module_results_df.to_csv(output_dir / "benchmarks.csv")
