from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy as np


def random_choice_with_seed(choices, num_choices, seed):
    if num_choices < 1 or seed < 1:
        clients_ids = choices
    else:
        if seed is not None:
            rs = RandomState(MT19937(SeedSequence(seed)))
            clients_ids = rs.choice(choices, size=num_choices, replace=False)
        else:
            clients_ids = np.random.choice(choices, size=num_choices, replace=False)

    return clients_ids
