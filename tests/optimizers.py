# Licensed under GLPv3 - see LICENSE
"""Define the optimizers to use in the benchmark."""

from sherpa.optmethods import LevMar, NelderMead, MonCar


def opt_default_LevMar():
    return LevMar()


def opt_LevMar_1e6():
    opt = LevMar(name="LevMar_1e-6")
    opt.ftol = 1e-6
    opt.xtol = 1e-6
    opt.gtol = 1e-6
    return opt


def opt_LevMar_1e5():
    opt = LevMar(name="LevMar_1e-5")
    opt.ftol = 1e-5
    opt.xtol = 1e-5
    opt.gtol = 1e-5
    return opt


def opt_LevMar_1e4():
    opt = LevMar(name="LevMar_1e-4")
    opt.ftol = 1e-4
    opt.xtol = 1e-4
    opt.gtol = 1e-4
    return opt


def opt_LevMar_1e3():
    opt = LevMar(name="LevMar_1e-3")
    opt.ftol = 1e-3
    opt.xtol = 1e-3
    opt.gtol = 1e-3
    return opt


def opt_default_NelderMead():
    return NelderMead()


def opt_default_MonCar():
    return MonCar()