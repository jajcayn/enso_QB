import numpy as np
import itertools as it
import copy


def get_continuous_differences(list_phases, points = 3):
    cont_phases = copy.deepcopy(list_phases)
    pairs = list(it.combinations(cont_phases, 2))
    avg_diff = np.zeros((len(pairs), cont_phases[0].shape[0],))
    for t in range(cont_phases[0].shape[0]):
        for cnt, pair in zip(range(len(pairs)), pairs):
            joint_min = np.amin([pair[0][t], pair[1][t]])
            # translate all to joint minimum
            for ph in pair:
                ph -= (ph[t] - joint_min)
            # compute avg diff over points
            avg_diff[cnt, t] = np.mean(np.abs(pair[0][max(t - points, 0) : min(t + points + 1, cont_phases[0].shape[0])] - pair[1][max(t - points, 0) : min(t + points + 1, cont_phases[0].shape[0])]))

    return np.mean(avg_diff, axis = 0)


def get_circular_differences(list_phases, points = 3):
    phases = copy.deepcopy(list_phases)
    pairs = list(it.combinations(phases, 2))
    avg_diff = np.zeros((len(pairs), phases[0].shape[0],))
    weights = np.concatenate([ np.linspace(0,1,points+2), np.linspace(0,1,points+1, endpoint = False)[::-1] ])[1:-1]
    for t in range(phases[0].shape[0]):
        for cnt, pair in zip(range(len(pairs)), pairs):
            arg = np.abs(pair[0][max(t - points, 0) : min(t + points + 1, phases[0].shape[0])] - 
                                                 pair[1][max(t - points, 0) : min(t + points + 1, phases[0].shape[0])])
            avg_diff[cnt, t] = np.average(arg, weights = weights[:arg.shape[0]])
#                                          weights = np.array(pascal(points*2)))

    return np.mean(avg_diff, axis = 0)



def synch_index_entropy(phase1, phase2, n, m, bins = 16):
    ph1 = phase1.copy()
    ph2 = phase2.copy()
    ph1 /= 2*np.pi
    ph2 /= 2*np.pi
    
    diff = n*ph1 - m*ph2

    phi = np.mod(diff, 1)
    
    S_max = np.log(bins)
#     bins_array = np.linspace(0, 1, bins+1)
    bins_array = np.linspace(phi.min(), phi.max(), bins+1)
#     print phi.min(), phi.max()
    S = 0
    for i in range(bins):
        ndx = (phi <= bins_array[i+1]) & (phi >= bins_array[i])
        pk = np.sum(ndx) / float(phi.shape[0])
        if pk != 0:
            S -= pk * np.log(pk)
    return (S_max - S) / S_max


def get_entropy_synch_index(list_phases, points = 3, bins = 8):
    phases = copy.deepcopy(list_phases)
    pairs = list(it.combinations(phases, 2))
    avg_diff = np.zeros((len(pairs), phases[0].shape[0],))
    for t in range(phases[0].shape[0]):
        for cnt, pair in zip(range(len(pairs)), pairs):
            avg_diff[cnt, t] = synch_index_entropy(pair[0][max(t - points,0) : min(t + points + 1, pair[0].shape[0])], pair[1][max(t - points,0) : min(t + points + 1, pair[1].shape[0])], 1, 1, bins = bins)

    return np.mean(avg_diff, axis = 0)


def get_circular_variance_index(list_phases, points = 3):
    phases = copy.deepcopy(list_phases)
    pairs = list(it.combinations(phases, 2))
    avg_diff = np.zeros((len(pairs), phases[0].shape[0],))
    for t in range(phases[0].shape[0]):
        for cnt, pair in zip(range(len(pairs)), pairs):
            avg_diff[cnt, t] = np.mean(np.exp(1j*np.abs(pair[0][max(t - points,0) : min(t + points + 1, pair[0].shape[0])] - pair[1][max(t - points,0) : min(t + points + 1, pair[1].shape[0])])))

    return np.mean(avg_diff, axis = 0)


def synch_index_strobo(phase1, phase2, n, m, no_bins = 4):
    ph1 = phase1.copy()
    ph2 = phase2.copy()
    
    ph1 = np.mod(ph1, 2*np.pi*m)
    ph2 = np.mod(ph2, 2*np.pi*n)

    lambda_bins = np.zeros((no_bins,))
    
    bins = np.linspace(0, 2*np.pi*m, no_bins + 1)
    for i in range(no_bins):
        ndx = (ph1 >= bins[i]) & (ph1 <= bins[i+1])
        lambda_bins[i] = np.nanmean(np.real([ np.exp(1j*(ph2[t] / float(n))) for t in np.where(ndx == True)[0]]))
        
    return np.nanmean(np.abs(lambda_bins))


def get_strobo_synch_index(list_phases, points = 3, bins = 16):
    import copy
    phases = copy.deepcopy(list_phases)
    pairs = list(it.combinations(phases, 2))
    avg_diff = np.zeros((len(pairs), phases[0].shape[0],))
    for t in range(phases[0].shape[0]):
        for cnt, pair in zip(range(len(pairs)), pairs):
            avg_diff[cnt, t] = synch_index_strobo(pair[0][max(t - points,0) : min(t + points + 1, pair[0].shape[0])], pair[1][max(t - points,0) : min(t + points + 1, pair[1].shape[0])], 1, 1, no_bins = bins)

    return np.mean(avg_diff, axis = 0)

        