# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 10:29:56 2020

@author: caitl
"""


from math import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from uncertainties import ufloat
from scipy import stats
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import quad
from scipy import signal
from scipy.interpolate import interp1d
import math
import cmath as cm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#from pandas import readcsv

directory =r'C:\Users\caitl\Documents\390\rclc'
os.chdir(directory)



df = pd.read_csv('Circuit1b.csv')
df = df.values
fspan = df[:,0]
gain = df[:,1]
phaseexp = df[:,2]
plt.figure(1)
plt.grid(1)
experimental=20*np.log10(gain)
plt.semilogx(fspan,experimental)
#plt.autoscale(tight=1)
plt.title("20logT")

plt.figure(2)
plt.grid(1)
plt.semilogx(fspan,phaseexp)
plt.autoscale(tight=1)
plt.title("Phase")


phaseexpd=phaseexp*np.pi/180
phasere=gain*np.cos(phaseexpd)
phaseim=gain*np.sin(phaseexpd)

#%%
R=1446.86
C=22*10**-9
f_0=5000
j=complex(0,1)
freq_range=np.arange(100,1000000)
ang_freq=freq_range*2*np.pi

T_a=R/(R+(1/(j*ang_freq*C)))
T=1-T_a
T_dB=20*np.log10(abs(T))
corner=20*np.log10(1/np.sqrt(2))
real=T.real
imag=T.imag

phase=np.zeros(len(freq_range))
for i in range(0,999900):
    phasei=math.degrees(math.atan(imag[i]/real[i]))
    phase[i]=phase[i]+phasei
    
fig,ax=plt.subplots()
ax.plot(freq_range,T_dB, label="Amplitude, Theoretical", color="navy")
ax.axhline(y=corner, color='navy', linestyle='dashed', label="$|T|_{dB}=1/\sqrt{2}$ ")
ax.set_ylabel("$|T|_{dB}$")
ax.set_xlabel("Frequency,Hz")
ax.grid()

ax2=ax.twinx()
ax2.semilogx(freq_range, phase, label="Phase, Theoretical")
ax2.axhline(y=-45, linestyle="dashed",label="$Phase=-45^o$")
ax2.axvline(x=f_0,  color="orange", label="Corner Frequency")
ax2.set_ylabel("Phase, degrees")

plt.title("Bode diagram (R="+str(R/1000)+"k$\Omega$,C="+str(round(C*10**9))+"nF)")
h,  l  = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h+h2, l+l2, loc=0)

plt.figure("Nyquist")
plt.plot(real,imag)
plt.title("Nyquist (R="+str(R/1000)+"k$\Omega$,C="+str(round(C*10**9))+"nF)")
plt.ylabel("Imaginary")
plt.xlabel("Real")
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()   

#%%

    
fig,ax=plt.subplots()
ax.plot(freq_range,T_dB, label="Theoretical", color="gray")
ax.plot(fspan,experimental, linestyle='dashed', color="navy",label="Amplitude, Multisim")
#ax.axhline(y=corner, color='navy', linestyle='dashed', label="$|T|_{dB}=1/\sqrt{2}$ ")
ax.set_ylabel("$|T|_{dB}$")
ax.set_xlabel("Frequency,Hz")
ax.grid()

ax2=ax.twinx()
ax2.semilogx(freq_range, phase,  color="gray")
ax2.semilogx(fspan,phaseexp,linestyle='dashed', label="Phase, Multisim")
#ax2.axhline(y=45, linestyle="dashed",label="$Phase=45^o$")
#ax2.axvline(x=f_0,  color="orange", label="Corner Frequency")
ax2.set_ylabel("Phase, degrees")

plt.title("Bode diagram (R="+str(R)+"$\Omega$,C="+str(round(C*10**9))+"nF)")
h,  l  = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h+h2, l+l2, loc=0)

plt.figure("Nyquist")
plt.plot(real,imag, color="grey", label="Theoretical")
plt.plot(phasere,phaseim,linestyle="dashed", label="Multisim")
plt.title("Nyquist (R="+str(R)+"$\Omega$,C="+str(round(C*10**9))+"nF)")
plt.ylabel("Imaginary")
plt.xlabel("Real")
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()   
plt.legend()
