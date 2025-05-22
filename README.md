# SOLPS-NN - A neural network model for the tokamak SOL plasma

## Requirements
The models have been trained and developed using the following package versions:
```
python 3.8.5
numpy 1.18.5
tensorflow 2.3.1
scikit-learn 0.23.2
matplotlib 3.3.1
```
The SOLPS-NN package should work fine also with later versions of the corresponding packages, but if problems occur it is recommended to fall back to these versions
The SOLPS-NN package should work on all operating systems, but has only been tested on Linux.

## Usage
### Importing the module
The package can be imported using `import solpsnn`
If the package folder is not directly adjacent to the calling python script, the path to the package can be temporarily added to the PYTHONPATH inside the script:
```
import sys
sys.path.append('path/to/solps-nn')
import solpsnn
```
### Model usage
The package offers primarily access to the `solpsnn.Model(item='te',species=None)` class.
The model can predict various  2D fields on the 104x50 computational grid:
- `item='te'` Electron temperature $\mathrm{eV}$
- `item='ti'` Ion temperature $\mathrm{eV}$

- `item='na',species='D0'` Deuterium neutral gas density $\mathrm{m}^{-3}$
- `item='na',species='D1'` Deuterium ion density $\mathrm{m}^{-3}$
- `item='na',species='N0'` Nitrogen neutral gas density $\mathrm{m}^{-3}$
- `item='na',species='N1'` Nitrogen $\mathrm{N}^{+1}$ ion density $\mathrm{m}^{-3}$
- `item='na',species='N2'` Nitrogen $\mathrm{N}^{+2}$ ion density $\mathrm{m}^{-3}$
- `item='na',species='N3'` Nitrogen $\mathrm{N}^{+3}$ ion density $\mathrm{m}^{-3}$
- `item='na',species='N4'` Nitrogen $\mathrm{N}^{+4}$ ion density $\mathrm{m}^{-3}$
- `item='na',species='N5'` Nitrogen $\mathrm{N}^{+5}$ ion density $\mathrm{m}^{-3}$
- `item='na',species='N6'` Nitrogen $\mathrm{N}^{+6}$ ion density $\mathrm{m}^{-3}$
- `item='na',species='N7'` Nitrogen $\mathrm{N}^{+7}$ ion density $\mathrm{m}^{-3}$

- `item='ua',species='D0'` Deuterium neutral gas parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='D1'` Deuterium ion parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N0'` Nitrogen neutral gas parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N1'` Nitrogen $\mathrm{N}^{+1}$ ion parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N2'` Nitrogen $\mathrm{N}^{+2}$ ion parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N3'` Nitrogen $\mathrm{N}^{+3}$ ion parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N4'` Nitrogen $\mathrm{N}^{+4}$ ion parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N5'` Nitrogen $\mathrm{N}^{+5}$ ion parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N6'` Nitrogen $\mathrm{N}^{+6}$ ion parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N7'` Nitrogen $\mathrm{N}^{+7}$ ion parallel velocity $\mathrm{m}/\mathrm{s}$

Additionally the model can predict the following scalar quantities
- `item='pwmxap'` Peak heatflux at the outer (LFS) divertor target $\mathrm{W}/\mathrm{m}^2$
- `item='fnixap'` Integrated deuterium ion flux to the outer (LFS) divertor target  $\mathrm{atoms}/\mathrm{s}$
- `item='psol'` Power crossing the separtrix from core to SOL plasma $\mathrm{W}$

To perform predictions after the model has been initialized the `Model.predict` method can be called.
The method expects as input a numpy.array of shape (N,8). Here N>=1 specifies the number of predictions that are made by the model. Especially when executing the model on GPUs it is way faster to execute multi predictions at once, in contrast to multiple calls of the models  `predict` method.

For each prediction the eight input parameters need to be given in that order in the given units:
- $R$ Tokamak major radius $\mathrm{m}$
- $B$ Toroidal magnetic field strength (on axis) $\mathrm{T}$
- $P$ Input power into simulation domain $\mathrm{W}$
- $D_\mathrm{puff}$ Deuterium gas puff rate $\mathrm{atoms}/\mathrm{s}$
- $N_\mathrm{puff}$ Nitrogen gas puff rate $\mathrm{atoms}/\mathrm{s}$
- $D_\mathrm{core}$ Deuterium core fueling rate $\mathrm{atoms}/\mathrm{s}$
- $D_\perp$ Cross-field particle transport coefficient $\mathrm{m}^2/\mathrm{s}$
- $\chi_\perp$ Cross-field heat transport coefficient  $\mathrm{m}^2/\mathrm{s}$

See the example.py script for setting the model parameters.

### Model download
Before the model can be used the neural network weights have to be downloaded.
To download a model invoke: `solpsnn.download_model(item='te',species=None)`.
To download all available models use `item='all'`.
If all models are downloaded more than 20GB of disk space are required.
### Parameter range
The model has been trained with simulations spanning the following parameter ranges:

| Parameter         | Range                                      |
| ----------------- | ------------------------------------------ |
| $R$               | $1-10\mathrm{m}$                           |
| $B$               | $1-10\mathrm{T}$                           |
| $P$               | $10-200\mathrm{MW}$                        |
| $D_\mathrm{puff}$ | $10^{18}-10^{24}\mathrm{atoms}/\mathrm{s}$ |
| $N_\mathrm{puff}$ | $10^{18}-10^{23}\mathrm{atoms}/\mathrm{s}$ |
| $D_\mathrm{core}$ | $10^{19}-10^{24}\mathrm{atoms}/\mathrm{s}$ |
| $D_\perp$         | $0.1-2\mathrm{m}^2/\mathrm{s}$             |
| $\chi_\perp$      | $0.1-2\mathrm{m}^2/\mathrm{s}$             |

Using the model to make predictions outside of these ranges is possible but not recommended.
It is doubtful whether the model predictions for such extrapolations contain any physical meaning.
### Memory managment
The neural networks loaded in each model are quite large, such that loading multiple models might exceed the systems memory capacity.
To circumvent this the predictions for the different plasma properties should be made one after another such, that only one model is loaded at the time.
Before loading a new model it is recommended to invoke `keras.backend.clear_session()` to remove all network layers still occupying the keras namespace and system memory.
### Model accuracy
The model has been trained using a database of fluid neutral SOLPS-ITER simulations.
This reduced fidelity simulation setup introduces unphysical simulation artifact into the model here
The most noticeable is the occurance of large temperature spikes at the lower edge of the outer target in the private flux region. In these cases also the peak heatflux to the target might be unphysically high, as it occurs at this location.
## Acknowledgement
This package has been developed as part of the PhD Thesis of S. Dasbach conducted at the Forschungszentrum Jülich.
Further improvements are currently developed at the Dutch Institute for Fundamental Energy Research.

This work has been carried out within the framework of the EUROfusion Consortium,
funded by the European Union via the Euratom Research and Training
Programme (Grant Agreement No 101052200 — EUROfusion). Views and opinions
expressed are however those of the author(s) only and do not necessarily reflect
those of the European Union or the European Commission. Neither the European
Union nor the European Commission can be held responsible for them.

The author gratefully acknowledges computing time on the supercomputer JURECA at Forschungszentrum Jülich under grant no. solsur.
## License
SOLPS-NN is licensed under the MIT license.

Copyright (c) 2024 Stefan Dasbach (Forschungszentrum Jülich)
## Citation
Users of the model are kindly asked to cite all of the following publications:
- Dasbach, S., & Wiesen, S. (2023). Towards fast surrogate models for interpolation of tokamak edge plasmas. _Nuclear Materials and Energy_, _34_, 101396.
