import math

import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.io as pio
from neuron import gui, h
from neuron.units import ms, mV

pio.renderers.default = "browser"

h.load_file("stdrun.hoc")


class Cell:
    def __init__(self, gid, x, y, z):
        self._gid = gid
        self._setup_morphology()
        self.all = self.soma.wholetree()
        self._setup_biophysics()
        self.x = self.y = self.z = 0
        h.define_shape()
        self._set_position(x, y, z)
        self.soma_v = h.Vector().record(self.soma(0.5)._ref_v)

    def __repr__(self):
        return "{}[{}]".format(self.name, self._gid)

    def _set_position(self, x, y, z):
        for sec in self.all:
            for i in range(sec.n3d()):
                sec.pt3dchange(
                    i,
                    x - self.x + sec.x3d(i),
                    y - self.y + sec.y3d(i),
                    z - self.z,
                    sec.diam3d(i),
                )
        self.x, self.y, self.z = x, y, z


class PointNeuron(Cell):
    name = "PointNeuron"

    def _setup_morphology(self):
        self.soma = h.Section(name="soma", cell=self)
        self.soma.L = self.soma.diam = 10

    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 100  # Axial resistance in Ohm * cm
            sec.cm = 1  # Membrane capacitance in micro Farads / cm^2
        self.soma.insert("hh")
        for seg in self.soma:
            seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003  # Leak conductance in S/cm2
            seg.hh.el = -54.3  # Reversal potential in mV

        # NEW: the synapse
        self.syn_E = h.Exp2Syn(self.soma(0.5))
        self.syn_E.tau1 = 0.5 * ms
        self.syn_E.tau2 = 2 * ms
        self.syn_E.e = 0 * mV

        self.syn_I = h.Exp2Syn(self.soma(0.5))
        self.syn_I.tau1 = 0.5 * ms
        self.syn_I.tau2 = 2 * ms
        self.syn_I.e = -70 * mV

        self.syn_ = h.ExpSyn(self.soma(0.5))
        self.syn_.tau = 0.1 * ms
        self.syn_.e = 0 * mV

        self.eeg_E = h.Vector().record(self.syn_E._ref_i)
        self.eeg_I = h.Vector().record(self.syn_I._ref_i)
        self.eeg_ = h.Vector().record(self.syn_._ref_i)


def create_n_PointNeurons(NETDIM_X, NETDIM_Y, NETDIM_Z, x0, y0, z0):
    # NETDIM_X, NETDIM_Y, NETDIM_Z - number of cells in X, Y and Z dimension
    # x0, y0, z0 – network point of origin
    Dx = 50  # Lattice X constant (um)
    Dy = 50  # Lattice Y constant (um)
    Dz = 50  # Lattice Z constant (um)

    cells = []
    l = 0

    for i in range(NETDIM_X):
        for j in range(NETDIM_Y):
            for k in range(NETDIM_Z):
                cells.append(PointNeuron(l, x0 + i * Dx, y0 + j * Dy, z0 + k * Dz))
                l = l + 1
    return cells


# gorna warstwa: komorki pobudzajace
my_cells_E = create_n_PointNeurons(8, 8, 1, 0, 0, 0)

# dolna warstwa: komorki hamujace
my_cells_I = create_n_PointNeurons(4, 4, 1, 100, 100, -100)


# Kazda komorka pobudzajaca dostaje nieskorelowane, losowe wejscie zewnetrzne:
netcons_stim = []

stims = []

for i in range(len(my_cells_E)):
    stim = h.NetStim(0.5)

    stim.number = 100
    stim.interval = 15
    stim.start = 0
    stim.noise = 1  # 0 – deterministic, 1 – Poisson process
    ncstim = h.NetCon(stim, my_cells_E[i].syn_, 0.001, 0, 0.01)
    netcons_stim.append(ncstim)
    stims.append(stim)


netcons_E = []

CONNECTION_PROB_EE = 0.05  # prawdopodobienstwo polaczenia pobudzajaca - pobudzajaca
weight_ee = 0.0001
CONNECTION_PROB_II = 0.1  # prawdopodobienstwo polaczenia hamujaca - hamujaca
weight_ii = 0.0003
# CONNECTION_PROB_EI = 0   # prawdopodobienstwo polaczenia pobudzajaca - hamujaca
CONNECTION_PROB_EI = 0.3  # prawdopodobienstwo polaczenia pobudzajaca - hamujaca
weight_ei = 0.0002
# CONNECTION_PROB_IE = 0   # prawdopodobienstwo polaczenia hamujaca - pobudzajaca
CONNECTION_PROB_IE = 1  # prawdopodobienstwo polaczenia hamujaca - pobudzajaca
weight_ie = 0.003

# polaczenia komorek pobudzajaca - pobudzajaca  --> pobudzenie komorki pobudzajacej
for source in my_cells_E:
    for target in my_cells_E:
        prob = np.random.uniform()
        if (prob <= CONNECTION_PROB_EE) and (source != target):
            dist = math.dist(
                (source.x, source.y, source.z), (target.x, target.y, target.z)
            )
            nc_E = h.NetCon(source.soma(0.5)._ref_v, target.syn_E, sec=source.soma)
            nc_E.weight[0] = weight_ee
            nc_E.delay = 0.0005 * dist
            nc_E.threshold = 0
            netcons_E.append(nc_E)

# polaczenia komorek hamujaca - hamujaca  --> zahamowanie komorki hamujacej
for source in my_cells_I:
    for target in my_cells_I:
        prob = np.random.uniform()
        if (prob <= CONNECTION_PROB_II) and (source != target):
            dist = math.dist(
                (source.x, source.y, source.z), (target.x, target.y, target.z)
            )
            nc_E = h.NetCon(source.soma(0.5)._ref_v, target.syn_I, sec=source.soma)
            nc_E.weight[0] = weight_ii
            nc_E.delay = 0.0005 * dist
            nc_E.threshold = 0
            netcons_E.append(nc_E)

# polaczenia komorek pobudzajaca - hamujaca  --> pobudzenie komorki hamujacej
for source in my_cells_E:
    for target in my_cells_I:
        prob = np.random.uniform()
        if prob <= CONNECTION_PROB_EI:
            dist = math.dist(
                (source.x, source.y, source.z), (target.x, target.y, target.z)
            )
            nc_E = h.NetCon(source.soma(0.5)._ref_v, target.syn_E, sec=source.soma)
            nc_E.weight[0] = weight_ei
            nc_E.delay = 0.0005 * dist
            nc_E.threshold = 0
            netcons_E.append(nc_E)

# polaczenia komorek hamujaca - pobudzajaca  --> zahamowanie komorki pobudzajacej
for source in my_cells_I:
    for target in my_cells_E:
        prob = np.random.uniform()
        if prob <= CONNECTION_PROB_IE:
            dist = math.dist(
                (source.x, source.y, source.z), (target.x, target.y, target.z)
            )
            nc_E = h.NetCon(source.soma(0.5)._ref_v, target.syn_I, sec=source.soma)
            nc_E.weight[0] = weight_ie
            nc_E.delay = 0.0005 * dist
            nc_E.threshold = 0
            netcons_E.append(nc_E)


num_s = len(netcons_stim)
num_e = len(my_cells_E)

# recording stimulus
spike_times_stim = h.Vector()
idvec_stim = h.Vector()
for i, nc_stim in enumerate(netcons_stim):
    nc_stim.record(spike_times_stim, idvec_stim, i)

# recording spikes E
spike_times_vec_E = h.Vector()
idvec_E = h.Vector()
for i, cell in enumerate(my_cells_E):
    nc_E = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
    nc_E.record(spike_times_vec_E, idvec_E, i + num_s)
del nc_E

# recording spikes I
spike_times_vec_I = h.Vector()
idvec_I = h.Vector()
for i, cell in enumerate(my_cells_I):
    nc_I = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
    nc_I.record(spike_times_vec_I, idvec_I, i + num_s + num_e)
del nc_I

tstop = 200

t = h.Vector().record(h._ref_t)
h.finitialize(-65 * mV)
h.continuerun(tstop)

K = 1
plt.subplots(K, 2, sharex=True, sharey=True)
for i in range(K):
    plt.subplot(K, 2, 2 * i + 1)
    plt.plot(t, my_cells_E[i].soma_v, label=f"my_cells_E[{i}]")
    plt.ylabel("V (mV)")
    plt.xlabel("Time (ms)")
    plt.legend()

    plt.subplot(K, 2, 2 * i + 2)
    plt.plot(t, my_cells_I[i].soma_v, label=f"my_cells_I[{i}]")
    plt.xlabel("Time (ms)")
    plt.legend()
plt.show()


# plotting EEG signal
eeg = 0
for i in range(len(my_cells_E)):
    eeg += my_cells_E[i].eeg_E + my_cells_E[i].eeg_I + my_cells_E[i].eeg_

for i in range(len(my_cells_I)):
    eeg += my_cells_I[i].eeg_E + my_cells_I[i].eeg_I

eeg /= len(my_cells_E) + len(my_cells_I)

plt.figure()
plt.plot(t, eeg)
plt.ylabel("V (mV)")
plt.xlabel("Time (ms)")
plt.legend()
plt.show()

# plotting rasterplot
plt.figure()
for t, id in zip(spike_times_stim, idvec_stim):
    plt.vlines(t, id - num_e, id - (num_e - 1), colors="r")
for t, id in zip(spike_times_vec_E, idvec_E):
    plt.vlines(t, id - num_e, id - (num_e - 1), colors="b")
for t, id in zip(spike_times_vec_I, idvec_I):
    plt.vlines(t, id - num_e, id - (num_e - 1), colors="y")
plt.xlabel("Time (ms)")
plt.ylabel("Cell number")
plt.show()

# plotting histogram
plt.figure()
plt.hist(spike_times_vec_E, bins=tstop)
plt.hist(spike_times_vec_I, bins=tstop)
plt.show()
