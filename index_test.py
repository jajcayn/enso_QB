import numpy as np
import matplotlib.pyplot as plt
import matplotlib
cmap = matplotlib.cm.get_cmap('brg')

from datetime import date, datetime
import sys
sys.path.append('/Users/nikola/work-ui/multi-scale')
from src.data_class import DataField, load_enso_index, load_station_data
from scipy.signal import argrelextrema
import differences_functions as df


def plot_enso_episodes(min, max):
    nino_strong = [1957, 1965, 1972, 1982, 1997, 2015]
    for y in nino_strong:
        start = enso.find_date_ndx(date(y, 9, 1))
        end = enso.find_date_ndx(date(y+1, 5, 1))
        if start is None and end is not None:
            start = 0
        if end is None and start is not None:
            end = enso.data.shape[0]
        if start is None and end is None:
            continue
        x = np.arange(start, end, 1)
        plt.fill_between(x, min, max, facecolor = "#F83F67", edgecolor = "#F83F67", alpha = 0.7)

    nino_moderate = [1963, 1986, 1987, 1991, 2002, 2009]
    for y in nino_moderate:
        start = enso.find_date_ndx(date(y, 9, 1))
        end = enso.find_date_ndx(date(y+1, 5, 1))
        if start is None and end is not None:
            start = 0
        if end is None and start is not None:
            end = enso.data.shape[0]
        if start is None and end is None:
            continue
        x = np.arange(start, end, 1)
        plt.fill_between(x, min, max, facecolor = "#F83F67", edgecolor = "#F83F67", alpha = 0.3)

    nina_strong = [1973, 1975, 1988]
    for y in nina_strong:
        start = enso.find_date_ndx(date(y, 9, 1))
        end = enso.find_date_ndx(date(y+1, 5, 1))
        if start is None and end is not None:
            start = 0
        if end is None and start is not None:
            end = enso.data.shape[0]
        if start is None and end is None:
            continue
        x = np.arange(start, end, 1)
        plt.fill_between(x, min, max, facecolor = "#30AEDF", edgecolor = "#30AEDF", alpha = 0.7)
    
    nina_moderate = [1955, 1970, 1998, 1999, 2007, 2010]
    for y in nina_moderate:
        start = enso.find_date_ndx(date(y, 9, 1))
        end = enso.find_date_ndx(date(y+1, 5, 1))
        if start is None and end is not None:
            start = 0
        if end is None and start is not None:
            end = enso.data.shape[0]
        if start is None and end is None:
            continue
        x = np.arange(start, end, 1)
        plt.fill_between(x, min, max, facecolor = "#30AEDF", edgecolor = "#30AEDF", alpha = 0.3)


enso = load_enso_index("../data/nino34raw.txt", '3.4', date(1900, 1, 1), date(2015, 1, 1), anom = True)

def get_cycles_and_phases(periods):
    cycles = []
    cycles_param = []
    phases = []
    param_phases = []
    cont_phases = []
    cont_param_phases = []
    for period in periods:
        enso.wavelet(period, period_unit = 'm', cut = 12)
        amp = enso.amplitude[ndx[12:-12]].copy()
        enso.phase = enso.phase[ndx[12:-12]]
        phases.append(enso.phase)
        cycles.append(amp * np.cos(enso.phase))
        enso.wavelet(period, period_unit = 'm', cut = 12, continuous_phase = True)
        cont_phases.append(enso.phase[ndx[12:-12]])
        
        enso.get_parametric_phase(period, window = 5*12, period_unit = 'm', cut = 12)
        param_phases.append(enso.phase[ndx[12:-12]])
        cycles_param.append(amp * np.cos(enso.phase[ndx[12:-12]]))
        enso.get_parametric_phase(period, window = 5*12, period_unit = 'm', cut = 12, continuous_phase = True)
        cont_param_phases.append(enso.phase[ndx[12:-12]])
    return cycles, cycles_param, phases, cont_phases, param_phases, cont_param_phases


def mark_nino_periods(list_qb, cycles, ann, synch_function, comp_function, threshold, ratios = None, log = False, points = 3, 
    debug = False):
    # get synch index
    synch_idx = synch_function(list_qb, ratios = ratios, points = points)
    if log:
        synch_idx = np.log(synch_idx)
    assert ann.shape[0] == synch_idx.shape[0]

    # conditions:
    # -- synch_index is {< | >} than threshold
    # -- annual cycle component > 0
    # -- QB cycle component > 0
    ninos = np.zeros_like(ann)
    for t in range(ann.shape[0]):
        if comp_function(synch_idx[t], threshold) and ann[t] > 0 and cycles[0][t] > 0:
            ninos[t] = 1
        if debug:
            print t, synch_idx[t], comp_function(synch_idx[t], threshold), ann[t], cycles[0][t], ninos[t]

    return ninos




PERIODS = [12] + [i for i in range(20,27,2)]
# PERIODS = [12, 24]
# RATIOS = [1, 2]
# RATIOS = [1] + [2 for i in range(20,27,2)]
# RATIOS = None
RATIOS = [i/12. for i in PERIODS]
colors = [cmap(f) for f in np.linspace(0, 1, len(PERIODS))]
ndx = enso.select_date(date(1950, 1, 1), date(2010, 1, 1), apply_to_data = False)
cycles, cycles_param, phases, cont_phases, param_phases, cont_param_phases = get_cycles_and_phases(PERIODS)

enso.wavelet(12, period_unit = 'm', cut = 12)
enso.amplitude = enso.amplitude[ndx[12:-12]]
enso.phase = enso.phase[ndx[12:-12]]
enso_ann = enso.amplitude * np.cos(enso.phase)

enso.data = enso.data[ndx]
enso.time = enso.time[ndx]

TO = enso.get_date_from_ndx(-1).year
FROM = enso.get_date_from_ndx(0).year



thresh = 0.03
points = 10

plt.figure(figsize=(15,8))
# diff = df.get_gradient_index(phases, ratios = RATIOS, points = points)
diff = np.log(df.get_intermittent_gradient_index(phases, ratios = RATIOS, include_second_order = True))
p, = plt.plot(diff, label = "WVLT: mean gradient; points = %d" % points, color = 'k')
marked_ninos = mark_nino_periods(phases, cycles, enso_ann, synch_function = df.get_gradient_index, 
    comp_function = np.less_equal, threshold = thresh, log = False, points = points, ratios = RATIOS)
# for t in range(marked_ninos.shape[0]):
    # if marked_ninos[t] == 1:
        # plt.plot(t, diff[t], 'o', color = p.get_color(), markersize = 7)

plt.axhline(thresh, 0, 1, color = 'k', linewidth = 0.5)
# plt.plot(df.get_gradient_index(param_phases, ratios = RATIOS, points = points), '--', color = p.get_color(), label = "PARAM: mean gradient; points = %d" % points)
for cyc, col in zip(cycles, colors):
    plt.plot(cyc, color = col, linewidth = 1)
plot_enso_episodes(-4, 4)
plt.xlim([0, enso.data.shape[0]])
plt.xticks(np.arange(0, enso.data.shape[0]+1, ((TO-FROM)/10)*12), np.arange(FROM, TO+1, (TO-FROM)/10), rotation = 30)
# plt.legend()
# plt.title("MEAN GRADIENT INDEX", size = 20)
plt.show()
