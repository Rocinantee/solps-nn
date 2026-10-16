import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

import solpsnn

keras.backend.clear_session()

# Creating input parameters for ITER scenario
R=[6.2] # Major radius
B=[5.3] # Magnetic field strength
Tcore=[100*10**6] # 100MW input power
Dpuff=10**np.linspace(20,24,20) # Scan 20 deuterium gas puff values
Npuff=[10**20] # Fixed nitrogen gaspuff
Dcore=[9.1*10**21] # Fixed deuterium core fuelling
Dperp=[0.3] # Cross-field particle transport coefficient
Chi=[1] # Cross-field heat transport coefficient

# Creating input array with size (20,8)
X_input = np.array(np.meshgrid(R,B,Tcore,Dpuff,Npuff,Dcore,Dperp,Chi),dtype='float').T.reshape(-1,8)

# Making the model prediction
model_te = solpsnn.Model(item='te',species=None)
te_pred = model_te.predict(X_input)
tesepa = te_pred[:,102,25]
tesepm = te_pred[:,64,25]

model_D1 = solpsnn.Model(item='na',species='D1')
ne_pred = model_D1.predict(X_input)
nesepa = ne_pred[:,102,25]
nesepm = ne_pred[:,64,25]

# Plotting the results
fig,axs = plt.subplots(1,2,figsize=(7,3))
ax = axs[0]
ax.plot(X_input[:,3],tesepa,label='Outer target',color='blue')
ax.plot(X_input[:,3],tesepm,label='Outer midplane',color='orange')
ax.set_xlabel('Deuterium gas puff [1/s]')
ax.set_ylabel('Electron temperature [eV]')
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid()
ax.set_xlim(1.0*10**20,1.0*10**24)
ax.set_ylim(1,10**4)
ax.legend()

ax=axs[1]
ax.plot(X_input[:,3],nesepa,label='Outer target',color='blue')
ax.plot(X_input[:,3],nesepm,label='Outer midplane',color='orange')
ax.set_xlabel('Deuterium gas puff [1/s]')
ax.set_ylabel(r'Deuterium ion density [$\mathrm{m}^{-3}$]')
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid()
ax.set_xlim(1.0*10**20,1.0*10**24)
ax.legend()

plt.tight_layout()
plt.savefig('example_local.png',dpi=500)
#plt.show()
