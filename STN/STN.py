import matplotlib.pyplot as plt
from neuron import h
from neuron.units import ms, mV

h.load_file("stdrun.hoc")

soma = h.Section(name="soma")
soma.nseg = 1
soma.L = 18.8
soma.diam = 18.8
soma.Ra = 123
soma.insert("hh")
for seg in soma:
    seg.ena = 71.5
    seg.ek = -89.1
    seg.hh.gnabar = 0.25
    seg.hh.gl = 0.0001666
    seg.hh.el = -60
# soma.insert("CaT")
# for seg in soma:
#     seg.eca = 126.1

iclamp = h.IClamp(soma(0.5))
iclamp.delay = 500
iclamp.dur = 500
iclamp.amp = -0.05

v = h.Vector().record(soma(0.5)._ref_v)  # Membrane potential vector
t = h.Vector().record(h._ref_t)  # Time stamp vector

h.finitialize(-65 * mV)
h.continuerun(2000 * ms)

plt.plot(t, v)
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.show()
