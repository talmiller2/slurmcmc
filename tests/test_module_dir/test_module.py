def fun_with_extra_arg(x, setup_dict):
    if setup_dict['weather'] == 'sunny':
        return x ** 2
    else:
        return x + 30
