def train_test_val_split(environments):
    """ Splits the environments into i.i.d training, validation and test sets.
    Utilizes the generate_synth_state_actions before splitting

    :param environments:
    :return: tuple (train, validation, test)
    """
    pass  # TODO


def generate_synth_state_actions(environments):
    """Generates synthetic state actions pairs for each environment based on a optimal calculated path,
    for each indivual environment

    :param environments: array of np.uint8 arrays
    :rtype: list of state action pairs
    :returns: a list of list with state action pairs e.g. environment with agent on pos 5 as state and move-right as action
    """
    pass  # TODO


def calculate_optimal_trajectory(environment):
    """
    Calculates the optimal trajectory to be used for the given environment
    :param environment:
    :return: either a list of the actions or an array with the agent at every optimal position
    """
    pass  # TODO
