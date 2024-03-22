from itertools import chain

import matplotlib.pyplot as plt
import numpy
from neuron import gui, h
from neuron.units import ms, mV

h.load_file("stdrun.hoc")

# --------------------- Model specification ---------------------

# topology
soma = h.Section(name="soma")
apical = h.Section(name="apical")
basilar = h.Section(name="basilar")
axon = h.Section(name="axon")

apical.connect(soma)  # 0-end of apical section ia attached to 1-end of a soma section
basilar.connect(soma(0))
axon.connect(soma(0))

# geometry
soma.L = 30  # if no units are specified, NEURON assumes um
soma.diam = 30
soma.nseg = 1

apical.L = 600
apical.diam = 1
apical.nseg = 23

basilar.L = 200
basilar.diam = 2
basilar.nseg = 5

axon.L = 1000
axon.diam = 1
axon.nseg = 37

# biophysics
for sec in h.allsec():
    sec.Ra = 100
    sec.cm = 1

# To calculate number of segments use formula:
# (pg.29, Chapter 5 of the NEURON book):
# This makes use of the function lambda () included in standard NEURON library stdlib.hoc
# forall {nseg = int((L/(0.1*lambda_f(100))+0.9)/2)*2+1}
# forall {print nseg}
# func lambda_f() { // currently accessed section, $1 == frequency
# return 1e5*sqrt(diam/(4*PI*$1*Ra*cm))

# In Python:
# for sec in h.allsec():
#    length_constant = 1e5*numpy.sqrt(sec.diam/(4*numpy.pi*100*sec.Ra*sec.cm))
#    sec.nseg = int((sec.L/(0.1*length_constant)+0.9)/2)*2+1
#    print(sec, sec.nseg)

soma.insert("hh")

apical.insert("pas")

basilar.insert("pas")

for seg in chain(apical, basilar):  # iterator over two sections
    seg.pas.g = 0.0002
    seg.pas.e = -65

axon.insert("hh")

# --------------------- Instrumentation ---------------------
# Synaptic input
syn = h.AlphaSynapse(0.5, sec=soma)
syn.onset = 0.5
syn.gmax = 0.05
syn.tau = 0.1
syn.e = 0

# Recordings
v_soma = h.Vector().record(soma(0.5)._ref_v)  # Membrane potential vector
v_apical = h.Vector().record(apical(0.5)._ref_v)
v_axon = h.Vector().record(axon(0.5)._ref_v)
t = h.Vector().record(h._ref_t)  # Time stamp vector

# --------------------- Simulation control ---------------------

h.dt = 0.025
tstop = 10
v_init = -65

h.finitialize(v_init)
h.fcurrent()
h.continuerun(tstop)


# cwiczenie 1
# print(len(t))

plt.figure(figsize=(12.8, 6.4))
for n in range(8):
    h.finitialize(v_init)
    h.continuerun(n)
    rvp = h.RangeVarPlot("v", axon(1), apical(1))
    my_plot = rvp.plot(plt, label=f"t={n}")

plt.legend()
plt.show()


# cwiczenie 2 - krok calkowania
#
DT = [1, 0.2, 0.1, 0.05, 0.025]
plt.figure()
for x in DT:
    h.dt = x
    h.finitialize(v_init)
    h.fcurrent()
    h.continuerun(tstop)
    plt.subplot(1, 3, 1).set_title("apical")
    plt.plot(t, v_apical, label=f"dt = {x}")
    plt.ylim(-80, 80)
    plt.subplot(1, 3, 2).set_title("soma")
    plt.plot(t, v_soma, label=f"dt = {x}")
    plt.ylim(-80, 80)
    plt.subplot(1, 3, 3).set_title("axon")
    plt.plot(t, v_axon, label=f"dt = {x}")
    plt.ylim(-80, 80)
plt.legend()
plt.show()

# cwiczenie 3
dlamb = [1, 0.3, 0.2, 0.1, 0.05]


plt.figure()
for d_lambda in dlamb:
    axon.L = 1000
    axon.diam = 1
    axon.Ra = 100
    axon.cm = 1

    for sec in h.allsec():
        length_constant = 1e5 * numpy.sqrt(
            sec.diam / (4 * numpy.pi * 100 * sec.Ra * sec.cm)
        )
        sec.nseg = int((sec.L / (d_lambda * length_constant) + 0.9) / 2) * 2 + 1

    h.finitialize(v_init)
    h.fcurrent()
    h.continuerun(tstop)
    plt.subplot(1, 3, 1).set_title("apical")
    plt.plot(t, v_apical, label=f"d_lambda = {d_lambda}")
    plt.ylim(-80, 80)
    plt.subplot(1, 3, 2).set_title("soma")
    plt.plot(t, v_soma, label=f"d_lambda = {d_lambda}")
    plt.ylim(-80, 80)
    plt.subplot(1, 3, 3).set_title("axon")
    plt.plot(t, v_axon, label=f"d_lambda = {d_lambda}")
    plt.ylim(-80, 80)

plt.legend()
plt.show()


# cwiczenie 4
dlamb = [1, 0.3, 0.2, 0.1, 0.05]

plt.figure()
for d_lambda in dlamb:

    for sec in h.allsec():
        length_constant = 1e5 * numpy.sqrt(
            sec.diam / (4 * numpy.pi * 100 * sec.Ra * sec.cm)
        )
        sec.nseg = int((sec.L / (d_lambda * length_constant) + 0.9) / 2) * 2 + 1

    h.finitialize(v_init)
    h.fcurrent()
    h.continuerun(1.5)
    rvp = h.RangeVarPlot("v", axon(1), apical(1))
    my_plot = rvp.plot(plt, label=f"dl={d_lambda}")

plt.title("Space plot = 1.5 ms")
plt.legend()
plt.show()
