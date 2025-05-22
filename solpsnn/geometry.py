import os
import sys
import copy

import numpy as np

# The path to the package directory
package_dir = os.path.dirname(os.path.realpath(__file__))

def read_b25formfile(file_path,field):
    found = False
    data = []
    n = None
    with open(file_path, 'r', buffering=100000, encoding='UTF-8') as src:
        for line in src:
            if len(data) == n:
                break
            if len(line) == 0:
                break

            if field in line and '*cf:' in line and not found:
                s = line.split()
                t = s[1]
                n = int(s[2])
                feature = s[3]
                found = True
            elif found:
                if t == 'char':
                    data = str(line[0:n])
                    break
                elif t == 'int':
                    s = line.split()
                    for v in s:
                        data.append(int(v))
                        continue
                elif t == 'real':
                    s = line.split()
                    for v in s:
                        data.append(float(v))
                        continue

    return np.array(data)

class Geometry:
    def __init__(self, b2fgmtry):
        # Load standard parameters
        self.nx,self.ny = read_b25formfile(b2fgmtry, ' nx,ny ')
        self.crx = read_b25formfile(b2fgmtry, ' crx ').reshape((self.nx + 2, self.ny + 2, 4), order='F')
        self.cry = read_b25formfile(b2fgmtry, ' cry ').reshape((self.nx + 2, self.ny + 2, 4), order='F')
        self.vol = read_b25formfile(b2fgmtry, ' vol ').reshape((self.nx + 2, self.ny + 2), order='F')
        self.rightcut = read_b25formfile(b2fgmtry, ' rightcut ')[0]
        self.leftcut = read_b25formfile(b2fgmtry, ' leftcut ')[0]
        self.topcut = read_b25formfile(b2fgmtry, ' topcut ')[0]
        self.jxa = int(np.floor(self.rightcut - (self.rightcut - self.leftcut) / 4))
        self.jxi = int(np.floor(self.leftcut + (self.rightcut - self.leftcut) / 4))
        self.iyref = self.topcut
        self.jsep = int(np.floor(self.topcut - 1))

class GeometryModel:
    def __init__(self, item=None, squeeze=False, custom_geometry_path=None):
        if custom_geometry_path == None:
            self.geo_JET = Geometry(os.path.join(package_dir,'geometry_data','b2fgmtry'))
        else:
            self.geo_JET = Geometry(custom_geometry_path)
        self.R_JET = 3.000727179161820
        self.item = item
        self.squeeze = squeeze
    def predict(self, X, item=None, squeeze=False):
        # Check if a prediction target has been specified
        if item == None:
            if self.item == None:
                raise ValueError('No prediction item specified.')
            else:
                item = self.item
        # Check if shallow mode is possible
        if squeeze or self.squeeze:
            # Check if all inputs use same tokamak size
            if np.all(X[:,0]==X[0,0]):
                X = X[0:1,0:1]
            else:
                print('Warning: Different tokamak sizes used in prediction.')
        R_scale = X[:,0]/self.R_JET
        if item == 'crx':
            return self.geo_JET.crx[np.newaxis, :, :, :] * R_scale[:, np.newaxis, np.newaxis, np.newaxis]
        elif item == 'cry':
            return self.geo_JET.cry[np.newaxis, :, :, :] * R_scale[:, np.newaxis, np.newaxis, np.newaxis]
        elif item == 'vol':
            return self.geo_JET.vol[np.newaxis, :, :] * (R_scale[:, np.newaxis, np.newaxis])**3
        else:
            raise ValueError(f'{item} is not a valid geometry target')

class B2GridConverter:
    """
    Map 2D arrays from one B2 grid to another.
    """
    def __init__(self,geometry_old,geometry_new,algorithm='distance_agnostic'):
        if algorithm != 'distance_agnostic':
            raise ValueError('Only distance_agnostic remapping is implemented.')
        self.geo_old = copy.deepcopy(geometry_old)
        self.geo_new = copy.deepcopy(geometry_new)
        new_index = np.array(np.meshgrid(np.arange(0,self.geo_new.nx+2),np.arange(0,self.geo_new.ny+2))).T
        self.transform_index = np.zeros(new_index.shape,dtype='int')
        for i in range(0,2):
            # Loop through inside and outside separatrix
            yl_new,yr_new = [[0,self.geo_new.topcut],[self.geo_new.topcut+1,self.geo_new.ny+1]][i]
            yl_old,yr_old = [[0,self.geo_old.topcut],[self.geo_old.topcut+1,self.geo_old.ny+1]][i]
            for j in range(0,3):
                # Loop through the 3 poloidal section
                xl_new,xr_new = [[0,self.geo_new.leftcut],[self.geo_new.leftcut+1,self.geo_new.rightcut],[self.geo_new.rightcut+1,self.geo_new.nx+1]][j]
                xl_old,xr_old = [[0,self.geo_old.leftcut],[self.geo_old.leftcut+1,self.geo_old.rightcut],[self.geo_old.rightcut+1,self.geo_old.nx+1]][j]
                # Selecting the section of indices we want to use
                ix_new = new_index[xl_new:xr_new+1,yl_new:yr_new+1,0]
                iy_new = new_index[xl_new:xr_new+1,yl_new:yr_new+1,1]
                # Performing the calculation
                s = (ix_new-xl_new)/(xr_new-xl_new)
                ix_old = xl_old + (xr_old-xl_old)*s
                s = (iy_new-yl_new)/(yr_new-yl_new)
                iy_old = yl_old + (yr_old-yl_old)*s
                # Saving the new indices in the correct location
                self.transform_index[xl_new:xr_new+1,yl_new:yr_new+1,0] = np.round(ix_old).astype('int')
                self.transform_index[xl_new:xr_new+1,yl_new:yr_new+1,1] = np.round(iy_old).astype('int')
    def _apply(self,input2D):
        """
        Map a 2D array onto the new B2 grid.
        """
        if input2D.shape[1:3] == (self.geo_old.nx+2,self.geo_old.ny+2):
            return input2D[:,self.transform_index[:,:,0],self.transform_index[:,:,1]]
        else:
            raise ValueError('2D input has wrong number of grid cells.')
    def apply(self,input2D,volume_rescale=False):
        """
        Map a 2D array onto the new B2 grid.
        
        Some scalar quantities like the absolute radiation (W)
        are not invariant to changes in cell volume.
        
        volume_rescale=True: Rescale outputs proportional to change in cell volume.
        """
        if volume_rescale:
            rescaling_factor = (np.expand_dims(self.geo_new.vol,axis=0)/self._apply(np.expand_dims(self.geo_old.vol,axis=0)))
            return self._apply(input2D)*rescaling_factor
        else:
            return self._apply(input2D)