import sbibm
from sbibm.tasks.fwd_only_example.task import FWD_ONLY


def test_task_constructs():

    t = FWD_ONLY()

    assert True


def test_obtain_task():

    task = sbibm.get_task("forward-only")

    assert task is not None


def test_obtain_prior():

    task = sbibm.get_task("forward-only")  # See sbibm.get_available_tasks() for all tasks
    prior = task.get_prior()

    assert prior is not None



def test_obtain_simulator():

    task = sbibm.get_task("forward-only")

    simulator = task.get_simulator()

    assert simulator is not None



def test_obtain_observe_once():

    task = sbibm.get_task("forward-only")

    x_o = task.get_observation(num_observation=1)

    assert x_o is not None
    assert hasattr(x_o, "shape")
    print(x_o.shape,x_o)



def test_obtain_prior_samples():

    task = sbibm.get_task("forward-only")
    prior = task.get_prior()
    nsamples = 10

    thetas = prior(num_samples=nsamples)

    assert thetas.shape == (nsamples,4)


def test_simulate_from_thetas():

    task = sbibm.get_task("forward-only")
    prior = task.get_prior()
    sim = task.get_simulator()
    nsamples = 10

    thetas = prior(num_samples=nsamples)
    xs = sim(thetas)

    assert xs.shape == (nsamples, 200)
