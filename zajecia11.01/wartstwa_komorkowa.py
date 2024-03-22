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

        # everything below here in this method is NEW
        self._spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_times = h.Vector()
        self._spike_detector.record(self.spike_times)

        self._ncs = []

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
        self.syn_E = h.ExpSyn(self.soma(0.5))
        self.syn_E.tau = 2 * ms


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


my_cells_E = create_n_PointNeurons(5, 5, 1, 0, 0, 0)

# Make a new stimulator and attach it to a synapse
# of the first cell in the network.
stim = h.NetStim()
syn_ = h.ExpSyn(my_cells_E[0].soma(0.5))
syn_.tau = 2 * ms
syn_.e = 0 * mV

stim.number = 100
stim.interval = 10
stim.start = 0
stim.noise = 1  # 0 – deterministic, 1 – Poisson process
ncstim = h.NetCon(stim, syn_, 0, 1, 0.005)

netcons_E = []
connections = []
CONNECTION_PROB = 0.2

axon = h.Section(name="axon")
axon.diam = 1

for source in my_cells_E:
    for target in my_cells_E:
        prob = np.random.uniform()
        if (prob <= CONNECTION_PROB) and (source != target):
            dist = math.dist((source.x, source.y), (target.x, target.y))
            nc_E = h.NetCon(source.soma(0.5)._ref_v, target.syn_E, sec=source.soma)
            nc_E.weight[0] = 0.005
            nc_E.delay = 0.0005 * dist  # 0.0005 [ms/um] -> 2 [m/s]
            netcons_E.append(nc_E)
            # # axon.pt3dadd(target.x, target.y, source.x, source.y)
            axon.pt3dadd(target.x + source.soma.diam / 2, target.y, target.z, axon.diam)
            axon.pt3dadd(source.x + source.soma.diam / 2, source.y, source.z, axon.diam)


ps = h.PlotShape(True)
ps.variable("v")
ps.scale(-80, 40)
ps.exec_menu("Shape Plot")
ps.exec_menu("Show Diam")
ps.show(0)
h.flush_list.append(ps)

# recording spikes
spike_times_vec_E = h.Vector()
idvec_E = h.Vector()
for i, cell in enumerate(my_cells_E):
    nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
    nc.record(spike_times_vec_E, idvec_E, i)
del nc

# recording stimulus
spike_times_stim = h.Vector()
idvec_stim = h.Vector()
ncstim.record(spike_times_stim, idvec_stim, 0)

t = h.Vector().record(h._ref_t)
h.finitialize(-65 * mV)
h.continuerun(200)

plt.figure()
plt.plot(t, my_cells_E[0].soma_v, label="my_cells_E[0]")
plt.xlabel("Time (ms)")
plt.ylabel("V (mV)")
plt.legend()
plt.show()

h.tstop = 200
h.nrncontrolmenu()
h.movierunpanel()

# plotting rasterplot
plt.figure()
for t, id in zip(spike_times_vec_E, idvec_E):
    plt.vlines(t, id + 0.5, id + 1.5, colors="b")
for t, id in zip(spike_times_stim, idvec_stim):
    plt.vlines(t, id - 0.5, id + 0.5, colors="r")
plt.xlabel("Time (ms)")
plt.ylabel("Cell number")
plt.show()

ps.plot(plotly).show()
