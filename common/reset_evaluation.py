import numpy as np
from numba import njit


def make_reset_kernel(trainer):
    encode = trainer.native_encode
    chunk = trainer.native_chunk

    @njit(cache=False)
    def evaluate(x, begin, end, horizon, mean, std, weights):
        output = np.empty((end - begin, 22), np.float64)
        for first in range(begin, end, horizon):
            last = min(first + horizon, end)
            previous = x[first].copy()
            current = x[first + 1].copy()
            memory = encode(current, weights[0], weights[1])
            values, previous, current, memory = chunk(x[first + 1:last + 1], previous, current, memory, True, mean, std, *weights)
            output[first - begin:last - begin] = values
        return output
    return evaluate


def make_physical_horizon_kernel(trainer, state_indices):
    step = trainer.native_step
    encode = trainer.native_encode
    si = np.asarray(state_indices, np.int64)

    @njit(cache=False)
    def rollout(previous_physical, current_physical, commands, mean, std, weights):
        previous = ((previous_physical - mean) / std).astype(np.float32)
        current = ((current_physical - mean) / std).astype(np.float32)
        memory = encode(current, weights[0], weights[1])
        output = np.empty((len(commands), 22), np.float64)
        for k in range(len(commands)):
            current[0] = (commands[k, 0] - mean[0]) / std[0]
            current[16] = (commands[k, 1] - mean[16]) / std[16]
            state, memory = step(previous, current, memory, *weights)
            previous = current.copy()
            for j in range(22):
                current[si[j]] = state[j]
                output[k, j] = np.float64(state[j]) * std[si[j]] + mean[si[j]]
        return output
    return rollout
