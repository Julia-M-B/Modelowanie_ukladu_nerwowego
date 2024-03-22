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

soma.insert("CaT")
soma.insert("htc")

for seg in soma:
    seg.eca = 126.1
    seg.CaT.gmax = 0.02
    seg.htc.ghbar = 0.0005


iclamp = h.IClamp(soma(0.5))
iclamp.delay = 700
iclamp.dur = 1600
iclamp.amp = -0.05

v = h.Vector().record(soma(0.5)._ref_v)  # Membrane potential vector
r = h.Vector().record(soma(0.5).CaT._ref_r)
s = h.Vector().record(soma(0.5).CaT._ref_s)
I_Ca = h.Vector().record(soma(0.5)._ref_ica)
Ih = h.Vector().record(soma(0.5).htc._ref_i)
t = h.Vector().record(h._ref_t)  # Time stamp vector

h.finitialize(-65 * mV)
h.continuerun(3000 * ms)

plt.subplot(4, 1, 1)
plt.plot(t, v)
# plt.xlabel("t (ms)")
plt.ylabel("V (mV)")

plt.subplot(4, 1, 2)
plt.plot(t, I_Ca)
# plt.xlabel("t (ms)")
plt.ylabel("I_Ca (mA/cm2)")

plt.subplot(4, 1, 3)
plt.plot(t, r)
plt.plot(t, s)
# plt.xlabel("t (ms)")
plt.ylabel("r, s gate")

plt.subplot(4, 1, 4)
plt.plot(t, Ih)
# plt.xlabel("t (ms)")
plt.ylabel("Ih (mA/cm2)")

plt.show()
