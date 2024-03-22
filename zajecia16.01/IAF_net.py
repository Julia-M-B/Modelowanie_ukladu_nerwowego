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
        self._setup_morphology(x, y, z)
        self._setup_biophysics()
        self._set_position(x, y, z)  # <-- NEW

    def __repr__(self):
        return "{}[{}]".format(self.name, self._gid)

    def _set_position(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class IAF_Neuron(Cell):
    name = "IAF_Neuron"

    def _setup_morphology(self, x, y, z):
        self.soma = h.Section(name="soma", cell=self)
        self.soma.diam = 10
        self.soma.pt3dclear
        self.soma.pt3dadd(x, y, z, 10)
        self.soma.pt3dadd(x + 10, y, z, 10)

    def _setup_biophysics(self):
        self.IAF = h.IntFire1()
        self.IAF.tau = 20
        self.IAF.refrac = 5


def create_n_IAF_Neurons(NETDIM_X, NETDIM_Y, NETDIM_Z, x0, y0, z0):
    Dx = 50
    Dy = 50
    Dz = 50
    cells = []
    l = 0
    for i in range(NETDIM_X):
        for j in range(NETDIM_Y):
            for k in range(NETDIM_Z):
                cells.append(IAF_Neuron(l, x0 + i * Dx, y0 + j * Dy, z0 + k * Dz))
                l = l + 1
    return cells


# gorna warstwa: komorki pobudzajace
my_cells_E = create_n_IAF_Neurons(8, 8, 1, 0, 0, 0)

# dolna warstwa: komorki hamujace
my_cells_I = create_n_IAF_Neurons(4, 4, 1, 100, 100, -100)

# Kazda komorka pobudzajaca dostaje nieskorelowane, losowe wejscie zewnetrzne:
for i in range(len(my_cells_E)):
    stim = h.NetStim()
    stim.number = 100
    stim.interval = 5
    stim.start = 0
    stim.noise = 1  # 0 - deterministic, 1 - Poisson process
    ncstim = h.NetCon(stim, my_cells_E[i].IAF, 0, 0, 0.3)

netcons_E = []
CONNECTION_PROB_EE = 0.6  # prawdopodobienstwo polaczenia pobudzajaca - pobudzajaca
weight_ee = 0.7
CONNECTION_PROB_II = 0.2  # prawdopodobienstwo polaczenia hamujaca - hamujaca
weight_ii = -0.1
CONNECTION_PROB_EI = 0.5  # prawdopodobienstwo polaczenia pobudzajaca - hamujaca
weight_ei = 0.4
CONNECTION_PROB_IE = 0.6  # prawdopodobienstwo polaczenia hamujaca - pobudzajaca
weight_ie = -0.7

# polaczenia komorek pobudzajaca - pobudzajaca
axon_ee = h.Section(name="axon")
axon_ee.diam = 1

for source in my_cells_E:
    for target in my_cells_E:
        prob = np.random.uniform()
        if (prob <= CONNECTION_PROB_EE) and (source != target):
            dist = math.dist((source.x, source.y), (target.x, target.y))
            nc_E = h.NetCon(source.IAF, target.IAF)
            nc_E.weight[0] = weight_ee
            nc_E.delay = 0.005 * dist
            nc_E.threshold = 0
            netcons_E.append(nc_E)
            axon_ee.pt3dadd(
                target.x + source.soma.diam / 2, target.y, target.z, axon_ee.diam
            )
            axon_ee.pt3dadd(
                source.x + source.soma.diam / 2, source.y, source.z, axon_ee.diam
            )

# polaczenia komorek hamujaca - hamujaca
axon_ii = h.Section(name="axon")
axon_ii.diam = 1

for source in my_cells_I:
    for target in my_cells_I:
        prob = np.random.uniform()
        if (prob <= CONNECTION_PROB_II) and (source != target):
            dist = math.dist((source.x, source.y), (target.x, target.y))
            nc_E = h.NetCon(source.IAF, target.IAF)
            nc_E.weight[0] = weight_ii
            nc_E.delay = 0.005 * dist
            nc_E.threshold = 0
            netcons_E.append(nc_E)
            axon_ii.pt3dadd(
                target.x + source.soma.diam / 2, target.y, target.z, axon_ii.diam
            )
            axon_ii.pt3dadd(
                source.x + source.soma.diam / 2, source.y, source.z, axon_ii.diam
            )

# polaczenia komorek pobudzajaca - hamujaca
axon_ei = h.Section(name="axon")
axon_ei.diam = 1

for source in my_cells_E:
    for target in my_cells_I:
        prob = np.random.uniform()
        if prob <= CONNECTION_PROB_EI:
            dist = math.dist(
                (source.x, source.y, source.z), (target.x, target.y, target.z)
            )
            nc_E = h.NetCon(source.IAF, target.IAF)
            nc_E.weight[0] = weight_ei
            nc_E.delay = 0.005 * dist
            nc_E.threshold = 0
            netcons_E.append(nc_E)
            axon_ei.pt3dadd(
                target.x + source.soma.diam / 2, target.y, target.z, axon_ei.diam
            )
            axon_ei.pt3dadd(
                source.x + source.soma.diam / 2, source.y, source.z, axon_ei.diam
            )

# polaczenia komorek hamujaca - pobudzajaca
axon_ie = h.Section(name="axon")
axon_ie.diam = 1

for source in my_cells_I:
    for target in my_cells_E:
        prob = np.random.uniform()
        if prob <= CONNECTION_PROB_IE:
            dist = math.dist(
                (source.x, source.y, source.z), (target.x, target.y, target.z)
            )
            nc_E = h.NetCon(source.IAF, target.IAF)
            nc_E.weight[0] = weight_ie
            nc_E.delay = 0.005 * dist
            nc_E.threshold = 0
            axon_ie.pt3dadd(
                target.x + source.soma.diam / 2, target.y, target.z, axon_ie.diam
            )
            axon_ie.pt3dadd(
                source.x + source.soma.diam / 2, source.y, source.z, axon_ie.diam
            )

mvec_E = h.Vector().record(my_cells_E[0].IAF._ref_m)
mvec_I = h.Vector().record(my_cells_I[0].IAF._ref_m)

ps = h.PlotShape(True)
ps.variable("v")
ps.scale(-80, 40)
ps.exec_menu("Shape Plot")
ps.exec_menu("Show Diam")
ps.show(0)
h.flush_list.append(ps)

ps.plot(plotly).show()

t = h.Vector().record(h._ref_t)
h.finitialize(-65 * mV)
h.continuerun(200 * ms)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(t, mvec_E, label="my_cells_E[0]")
plt.xlabel("Time (ms)")
plt.legend(loc="upper right")

plt.subplot(1, 2, 2)
plt.plot(t, mvec_I, label="my_cells_I[0]")
plt.xlabel("Time (ms)")
plt.legend(loc="upper right")
plt.show()
