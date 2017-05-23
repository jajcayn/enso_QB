import numpy as np
import sys
sys.path.append('/Users/nikola/work-ui/multi-scale')
from src.data_class import load_ERSST_data, load_enso_index, nandetrend
from datetime import date
import matplotlib.pyplot as plt
import scipy.stats as sts

sst = load_ERSST_data('/Users/nikola/work-ui/data/ersstv4/', date(1947, 10, 1), date(2016,3,1), None, None, False)
sst.data, _, _ = nandetrend(sst.data, axis = 0)
sst.anomalise(base_period = (date(1971, 1, 1), date(2001, 1, 1)))
sst.select_lat_lon([-5, 5], [190, 240], apply_to_data = True)
nino34 = np.nanmean(sst.data, axis = (1,2))
print nino34.shape

sst = load_ERSST_data('/Users/nikola/work-ui/data/ersstv4/', date(1947, 10, 1), date(2015,3,1), None, None, False)
sst.data, _, _ = nandetrend(sst.data, axis = 0)
sst.anomalise(base_period = (date(1971, 1, 1), date(2001, 1, 1)))
sst.select_lat_lon([0, 15], [280, 20], apply_to_data = True)
nta = np.nanmean(sst.data, axis = (1,2))
print nta.shape


enso = load_enso_index("/Users/nikola/work-ui/data/nino34raw.txt", '3.4', date(1946, 10, 1), date(2016, 3, 1), anom = False)
enso.wavelet(36, 'm', cut = 12, cut_time = False, cut_data = False, phase_fluct = False)
# enso.anomalise(base_period = (date(1971, 1, 1), date(2001, 1, 1)))
# ndx = enso.select_date(date(1947,10,1), date(2015, 3, 1), apply_to_data = False)

djf_ndx = sst.select_months([12, 1, 2], apply_to_data = False)
mam_ndx = sst.select_months([3, 4, 5], apply_to_data = False)

nino34 = nino34[djf_ndx].reshape(3, -1, order = "F").mean(axis = 0)
print nino34.shape

nta = nta[mam_ndx].reshape(3, -1, order = "F").mean(axis = 0)
print nta.shape

qb_amp = enso.amplitude[djf_ndx].reshape(3, -1, order = "F").mean(axis = 0)
# enso.get_parametric_phase(36, 24, 'm', cut = 12, cut_time = False, cut_data = False, phase_fluct = False)
plt.plot(enso.phase)
plt.show()
qb_ph_fluc = enso.phase[djf_ndx].reshape(3, -1, order = "F").mean(axis = 0)


# 21 year correlation window

subseq = False
percentil = 0.05
offset = 1 if subseq else 0
tit = "MAM NTA > DJF NINO" if subseq else "DJF NINO > MAM NTA"


correlation_n34_nta = []
correlation_n34_qb = []
correlation_nta_qb = []
correlation_nta_qb_ph = []
window = 21
for i in range(nta.shape[0] - window):
    # print nino34[i:i+window], nta[i:i+window]    
    n34, _, _ = nandetrend(nino34[offset+i:offset+i+window])
    nta_window, _, _ = nandetrend(nta[i:i+window])
    qb, _, _ = nandetrend(qb_amp[offset+i:offset+i+window])
    qb_ph, _, _ = nandetrend(qb_ph_fluc[offset+i:offset+i+window])
    if subseq:
        prev_n34, _, _ = nandetrend(nino34[i:i+window])
        # s, i, _, _, _ = sts.linregress(n34, prev_n34)
        # n34 -= s*np.arange(n34.shape[0]) + i
        # s, i, _, _, _ = sts.linregress(nta_window, prev_n34)
        # nta_window -= s*np.arange(nta_window.shape[0]) + i
    correlation_n34_nta.append(sts.pearsonr(n34, nta_window))
    correlation_n34_qb.append(sts.pearsonr(n34, qb))
    correlation_nta_qb.append(sts.pearsonr(nta_window, qb))
    correlation_nta_qb_ph.append(sts.pearsonr(nta_window, qb_ph))

correlation_n34_nta = np.array(correlation_n34_nta)
correlation_n34_qb = np.array(correlation_n34_qb)
correlation_nta_qb = np.array(correlation_nta_qb)
correlation_nta_qb_ph = np.array(correlation_nta_qb_ph)

p1, = plt.plot(correlation_n34_nta[:, 0], label = "NINO3.4 x NTA")
for i in range(correlation_n34_nta.shape[0]):
    if correlation_n34_nta[i, 1] <= percentil:
        plt.plot(i, correlation_n34_nta[i, 0], 'o', color = p1.get_color(), markersize = 7)
p1, = plt.plot(correlation_n34_qb[:, 0], label = "NINO3.4 x QB")
for i in range(correlation_n34_qb.shape[0]):
    if correlation_n34_qb[i, 1] <= percentil:
        plt.plot(i, correlation_n34_qb[i, 0], 'o', color = p1.get_color(), markersize = 7)
p1, = plt.plot(correlation_nta_qb[:, 0], label = "NTA x QB amp")
for i in range(correlation_nta_qb.shape[0]):
    if correlation_nta_qb[i, 1] <= percentil:
        plt.plot(i, correlation_nta_qb[i, 0], 'o', color = p1.get_color(), markersize = 7)
p1, = plt.plot(correlation_nta_qb_ph[:, 0], label = "NTA x QB phase flucts")
for i in range(correlation_nta_qb_ph.shape[0]):
    if correlation_nta_qb_ph[i, 1] <= percentil:
        plt.plot(i, correlation_nta_qb_ph[i, 0], 'o', color = p1.get_color(), markersize = 7)

plt.xticks(np.arange(0, correlation_n34_nta.shape[0], 5), np.arange(1958, 2016, 5), rotation = 30)
plt.legend()
plt.title(tit, size = 25)
plt.show()




