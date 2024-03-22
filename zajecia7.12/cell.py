import matplotlib.pyplot as plt
from neuron import gui, h
from neuron.units import ms, mV

h.load_file("stdrun.hoc")
# h.celsius = 37


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
    seg.CaT.gmax = 0.023
    seg.eca = 126.1
    seg.htc.eh = -47
    seg.htc.ghbar = 0.00026

iclamp = h.IClamp(soma(0.5))
iclamp.delay = 700
iclamp.dur = 1600
iclamp.amp = -0.05

v = h.Vector().record(soma(0.5)._ref_v)  # Membrane potential vector
t = h.Vector().record(h._ref_t)  # Time stamp vector
icat = h.Vector().record(soma(0.5)._ref_ica)  # Membrane potential vector
ih = h.Vector().record(soma(0.5)._ref_i_htc)  # Membrane potential vector
r = h.Vector().record(soma(0.5)._ref_r_CaT)  # Membrane potential vector
s = h.Vector().record(soma(0.5)._ref_s_CaT)  # Membrane potential vector


h.finitialize(-65 * mV)
h.continuerun(3000 * ms)

plt.figure(figsize=(6.4, 6.4))
plt.subplot(4, 1, 1)
plt.plot(t, v)
plt.ylim(-90, 80)
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")

plt.subplot(4, 1, 2)
plt.plot(t, icat)
plt.ylim(-0.5, 0.1)
plt.xlabel("t (ms)")
plt.ylabel("I_Ca (mA/cm2)")

plt.subplot(4, 1, 3)
plt.plot(t, r)
plt.ylim(0, 1)
plt.xlabel("t (ms)")
plt.ylabel("r gate)")

plt.subplot(4, 1, 3)
plt.plot(t, s)
plt.ylim(0, 1)
plt.xlabel("t (ms)")
plt.ylabel("r, s gate")

plt.subplot(4, 1, 4)
plt.plot(t, ih)
plt.ylim(-0.0002, 0.0004)
plt.xlabel("t (ms)")
plt.ylabel("Ih (mA/cm2)")

plt.show()
