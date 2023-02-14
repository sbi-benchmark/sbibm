import sbibm


def test_loading_ddm_task():

    sbibm.get_task("ddm")

def test_simulation_ddm_task():

    task = sbibm.get_task("ddm")
    prior = task.get_prior()
    simulator = task.get_simulator()
    simulator(prior(1))

