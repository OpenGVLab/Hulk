from ..solvers.solver_mae_devnew import TesterMAEDev

def tester_entry(C_train, C_test):
    return globals()[C_test.config['common']['tester']['type']](C_train, C_test)
