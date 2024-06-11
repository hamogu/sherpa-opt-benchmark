# Benchmark Sherpa Optimizers for speeds and accuracy

This repository contains a framework and some actual test cases for [Sherpa](https://sherpa.readthedocs.io). Unlike the unit test that are included in the Sherpa repository itself, here we collect long-running benchmarks for the numerical optimizers in Sherpa. The benchmark do not necessarily have pass/fail criteria, since different optimizers are good for different purposes. Instead, the benchmarks are intended to give a sense of the speed and accuracy of the optimizers in different situations.

The goal is to test the optimizers in typical use cases of sherpa, such as fitting X-ray spectra. Some benchmarks run a long time (> 15 min) and are thus not suitable for running in the CI system; instead, these benchmarks are intended to be run by developers on their own machines when developing or changing the algorithms of fits and optimizers.

In most cases, it won't be enough to simply run the benchmarks and compare the results. Developers will take them as starting point to look into particular use cases to understand how specific optimizers performed and what can be done about that.

## Design

While the benchmarks are not "tests" in the sense of traditional unit tests, the [pytest framework](https://docs.pytest.org/en/latest/) and its plug-ins provide a convenient way to run the benchmarks and collect the results. In particular, we use the [pytest-cases](https://smarie.github.io/python-pytest-cases/) plug-in for its increased flexibility to parameterize the test set-up compared to pytest itself and the [pytest-harvest](https://smarie.github.io/python-pytest-harvest) plug-in to collect test results.

Each "test" is a function that runs a particular optimization problem with a particular optimizer and records the time it took to run and the result of the optimization. The results are stored in a pandas dataframe, which can be saved to a file and loaded later for analysis.

Since some benchmarks run for a long time, we err on the side of caution and write out different outputs for every test to understand what happened. These are placed in the `results` directory
and include the pandas dataframe summary of all results, as well as log files, plots, and other outputs from the tests.


## Setting up an environment for run the benchmarks
Since this is for benchmarking to the run be developers on their own machines,
no attempt has been made to minimize dependencies.

This repository is intended to be cloned and then the benchmarks are run from the root of that cloned directory:

```bash
git clone git@github.com:hamogu/sherpa-opt-benchmark.git
cd sherpa-opt-benchmark
```

For the benchmarks, X-ray spectra are generated on the fly. The required ARF/RMF files for different instruments are taken from a repository of test that, which is included as a git submodule. To get the data, run the following commands:

```bash
git submodule init
```

The benchmarks require sherpa with XSPEC installed, see [the Sherpa installation instructions](https://sherpa.readthedocs.io/en/4.16.1/install.html). Other packages can be installed with pip:

```bash
pip install -r requirements.txt
```

If used to for profiling (see below) then the `graphviz` package is also required; this is best installed with a package manager, e.g. `apt-get install graphviz` on Ubuntu or with `conda`.

## Running the benchmarks
Since the benchmarks are orchestrated by pytest, they can be run with the following command:

```bash
pytest -v
```

### Running a subset of the benchmarks

Because all the text running is orchestrated by pytest, we can use the [pytest options such as -k to select tests](https://docs.pytest.org/en/latest/how-to/usage.html#specifying-tests-selecting-tests). For example, to run only the tests that have "MOS" or "synthesis" in their name, we can use the following command:

```bash
pytest -v -k "MOS or synthesis"
```
The "synthesis" is the "test" that summarizes all the test results into a pandas dataframe. So, if we want to see and save a summary of the test results, then we need to include it in the pytest command.

### Profiling
With the right pytest plug-ins behind the scences, we can also profile the tests:

```bash
pytest -v -k "acis_abspowerlaw-NelderMead" --profile-svg
```

This will create a `profile.svg` file in the `prof` directory, which can be opened in a web browser to understand where the time is being spent in the test.

### How do I add other optimizers or test cases?
The goal of this project is not to comprehensively test all possible problems that might be addressed with Sherpa, but to provide a few common cases to look at. We envision that a developer who wants to perform benchmarking for a particular optimizer or dataset would clone this repository and add that optimizer or dataset to the existing tests and then run it locally, looking at the results by eye, to see if it meets expectations.

## What we learn from benchmarking and profiling
The benchmarks show that the benchmark runtime correlates essentially linearly with `nfev` (number of function evaluations) that the optimizer does.
Running profiling on my laptop, I see that about 95% of the time is spent in the XSPEC binary code evaluating the model.

So, what we need to do to speed up sherpa is to

- use optimizers that converge in fewer steps,
- use caching so that the model functions are called less often,
- use the parallelization features of sherpa to evaluate the model in parallel.