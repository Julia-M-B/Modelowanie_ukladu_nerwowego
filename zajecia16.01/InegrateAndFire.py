import matplotlib.pyplot as plt
from neuron import gui, h
from neuron.units import ms, mV

h.load_file("stdrun.hoc")

IAF = h.IntFire1()
IAF.tau = 20  # time constant
IAF.refrac = 5  # refractory period

stim = h.NetStim()
stim.number = 25
stim.interval = 2
stim.start = 0
stim.noise = 1  # 0 - deterministic, 1 - Poisson process
ncstim = h.NetCon(stim, IAF, 0, 0, 0.3)

mvec = h.Vector().record(IAF._ref_m)

# modify advance() procedure in stdrun.hoc
h(
    """
proc advance() {
   fadvance()
   m1 = IntFire1[0].M()
}"""
)

Mvec = h.Vector().record(h._ref_m1)

soma = h.Section(name="soma")
t = h.Vector().record(h._ref_t)


# tworzenie sieci n sekwencyjnie polaczonych neuronow typu IAF
def create_n_IAF(n):
    cells = []
    for i in range(n):
        IAF_i = h.IntFire1()
        IAF_i.tau = 20  # time constant
        IAF_i.refrac = 2  # refractory period
        cells.append(IAF_i)
    return cells


my_cells = create_n_IAF(5)

# wejscie na pierwsza komorke
stim_net = h.NetStim()
stim_net.number = 25
stim_net.interval = 2
stim_net.start = 0
stim_net.noise = 1  # 0 - deterministic, 1 - Poisson process
ncstim_net = h.NetCon(stim_net, my_cells[0], 0, 0, 0.3)

connections = []

for i in range(len(my_cells) - 1):
    con_IAF = h.NetCon(my_cells[i], my_cells[i + 1], 0, 0, 0.3)
    connections.append(con_IAF)

# recording spikes
spike_times_vec_IAF = h.Vector()
idvec_IAF = h.Vector()
spike_times_vec_stim = h.Vector()
idvec_stim = h.Vector()
ncstim.record(spike_times_vec_stim, idvec_stim, 0)
nc = h.NetCon(IAF, None)
nc.record(spike_times_vec_IAF, idvec_IAF, 1)

h.finitialize(-65 * mV)
h.continuerun(100)

# plotowanie dla pojedynczej komorki
plt.figure()
plt.subplot(1, 3, 1)
plt.plot(t, mvec, label="m")
plt.xlabel("Time (ms)")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(t, Mvec, label="M")
plt.xlabel("Time (ms)")
plt.legend()

plt.subplot(1, 3, 3)
for t, id in zip(spike_times_vec_IAF, idvec_IAF):
    plt.vlines(t, id + 0.5, id + 1.5, colors="b")
for t, id in zip(spike_times_vec_stim, idvec_stim):
    plt.vlines(t, id - 0.5, id + 0.5, colors="r")
plt.xlabel("Time (ms)")
plt.show()

# SIEC KOMORKOWA

mvec_net_first = h.Vector().record(my_cells[0]._ref_m)
mvec_net_last = h.Vector().record(my_cells[len(my_cells) - 1]._ref_m)
# Mvec_net = h.Vector().record(h._ref_m1)
soma_net = h.Section(name="soma")
t_net = h.Vector().record(h._ref_t)

# recording spikes
spike_times_vec_stim_net = h.Vector()
idvec_stim_net = h.Vector()
ncstim_net.record(spike_times_vec_stim_net, idvec_stim_net, 0)

spike_times_vec_IAF_net = h.Vector()
idvec_IAF_net = h.Vector()
for i, IAF_cell in enumerate(my_cells):
    nc_net = h.NetCon(IAF_cell, None)
    nc_net.record(spike_times_vec_IAF_net, idvec_IAF_net, i)

h.finitialize(-65 * mV)
h.continuerun(200)

# plototwanie dla sieci
plt.figure()
plt.subplot(1, 3, 1)
plt.plot(t_net, mvec_net_first, label="IAF[0]")
plt.xlabel("Time (ms)")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(t_net, mvec_net_last, label="IAF[4]")
plt.xlabel("Time (ms)")
plt.legend()

plt.subplot(1, 3, 3)
for t, id in zip(spike_times_vec_IAF_net, idvec_IAF_net):
    plt.vlines(t, id + 0.5, id + 1.5, colors="b")
for t, id in zip(spike_times_vec_stim_net, idvec_stim_net):
    plt.vlines(t, id - 0.25, id + 0.25, colors="r")
plt.xlabel("Time (ms)")
plt.legend()

plt.show()
