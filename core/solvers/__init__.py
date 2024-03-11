from .solver import Solver
from .solver_deter import SolverDeter
from .solver_mae_devnew import SolverMAEDev

def solver_entry(C):
    return globals()[C.config['common']['solver']['type']](C)
