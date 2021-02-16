import logging
import autograd.numpy as np
from autograd import grad
from functools import partial


def cauchy_distribution(x, loc, scale):
    expr = 1 + ((x - loc) / scale) ** 2
    return 1 / (np.pi * scale * expr)


def standard_cauchy(x):
    loc = 0
    scale = 1
    return cauchy_distribution(x, loc, scale)

def fix_distribution_params(distribution, sample, fix_params):

    fixed_dist = partial(distribution, **fix_params)

    funcs = [
        partial(fixed_dist, observation)
        for observation in sample    
    ]

    return funcs


def likelihood(distribution, sample, fix_params):
    """
    Take a distribution, some sample data and kwargs representing
    the parameters of the distribution to fix and return the 
    likelihood function of the remaining free parameter
    (Only univariate problem supported)


    Args:
        distribution (type: function): function representing distribution
        sample (type: list): list of observations sampled from the distribution
        fix_params (type: dictionary): Dictionary of parameters to fix
    """

    funcs = fix_distribution_params(distribution, sample, fix_params)

    def _func(free_param):
        return np.prod([f(free_param) for f in funcs])

    return _func

def log_likelihood(distribution, sample, fix_params):
    """
    Take a distribution, some sample data and kwargs representing
    the parameters of the distribution to fix and return the 
    `LOG` likelihood function of the remaining free parameter
    (Only univariate problem supported)

    Args:
       distribution (type: function): function representing distribution
        sample (type: list): list of observations sampled from the distribution
        fix_params (type: dictionary): Dictionary of parameters to fix
    """
    
    funcs = fix_distribution_params(distribution, sample, fix_params)

    def _func(free_param):
        return np.sum([np.log(f(free_param)) for f in funcs])

    return _func


def newton_raphson(func, start_guess, max_iters, enable_logging=False):
    raphson_logger = logging.getLogger("Newton-Raphson")
    if enable_logging:
        raphson_logger.setLevel(logging.INFO)
    else:
        raphson_logger.setLevel(logging.FATAL)

    raphson_logger.info("\titeration\txn\tstep\tf(x)\tf\'(x)")
        
    if isinstance(start_guess, int):
        # https://github.com/HIPS/autograd/issues/482
        start_guess = float(start_guess)

    tol = 1e-6
    func_prime = grad(func)
    
    x = start_guess
    for iteration in range(0, max_iters):
        fx = func(x)
        fprime_x = func_prime(x)

        h = ( fx / fprime_x )
        raphson_logger.info(f"\t{iteration}\t{x}\t{h}\t{fx}\t{fprime_x}")
        
        if abs(h) < tol:
            break 

        x = x - h
    return x


# Tests 
if __name__ == "__main__":
    
    data = [9,9.5,11]
    
    def normal_distribution(x, mu, sig):
        # Normal distribution
        y = -1 * ( ((x - mu) ** 2) / (2 * (sig ** 2)) )
        z = np.exp(y)
        return z / (sig * np.sqrt(2 * np.pi))

    normal_likelihood = likelihood(
        normal_distribution,
        sample=data,
        fix_params={'sig': 1}
    )

    # Max value is around 9.83
    # Due to granularity of linspace, don't expect test algo to return true max mu value

    def get_max_likelihood(likelihood, param_range):

        param_values = np.linspace(param_range[0], param_range[1])
        param_distribution = list(map(normal_likelihood, param_values))

        # Get max in param distribution
        max_likelihood_idx = np.argmax(param_distribution)

        # max likelihood 
        max_likelihood = param_distribution[max_likelihood_idx]

        # max param value
        optimal_param = param_values[max_likelihood_idx]

        return (optimal_param, max_likelihood)


    optimal_mu, max_likelihood = get_max_likelihood(normal_likelihood, (8,12))

    expected_mu = 9.83
    assert abs(optimal_mu - expected_mu) < 0.1, "Mu value is off!"


