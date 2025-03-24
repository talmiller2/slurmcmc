def example_fun(x):
    return x ** 2


def example_fun_with_extra_arg(x, setup_dict):
    if setup_dict['weather'] == 'sunny':
        return x ** 2
    else:
        return x + 30


def example_constraint_fun(x):
    r_constraint = 3
    x0_constraint = -1
    y0_constraint = -1
    # return > 0 for violation
    if (x[0] - x0_constraint) ** 2 + (x[1] - y0_constraint) ** 2 > r_constraint ** 2:
        return 1
    else:
        return -1
