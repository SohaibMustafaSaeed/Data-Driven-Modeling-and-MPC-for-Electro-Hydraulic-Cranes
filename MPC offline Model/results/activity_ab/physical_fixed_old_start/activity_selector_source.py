def activity_segment(x, selection, mean, std, episode_steps, block):
    if isinstance(selection, (int, np.integer)):
        return int(selection)
    indices, last_start, stride = selection
    velocity = np.asarray(x[indices][:, MECH_INPUT[2:]], float) * std[MECH_INPUT[2:]] + mean[MECH_INPUT[2:]]
    window = max(1, int(math.ceil(episode_steps * block / stride)))
    if len(indices) < window:
        raise ValueError('Not enough rows for reference segment')
    rms = []
    for j in range(2):
        prefix = np.r_[0.0, np.cumsum(velocity[:, j] ** 2)]
        rms.append(np.sqrt(np.maximum((prefix[window:] - prefix[:-window]) / window, 0)))
    score = np.minimum(rms[0], rms[1]) + 0.1 * (rms[0] + rms[1])
    starts = indices[:len(score)]
    valid = starts <= last_start
    if not valid.any():
        raise ValueError('No complete automatic reference segment')
    return int(starts[valid][np.argmax(score[valid])])
