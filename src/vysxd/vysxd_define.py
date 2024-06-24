import tkinter #, tkinter.filedialog
import numpy as np
import h5py
import os
import time

# class vysxd_data_object(object):
# 	def __init__(self, h5_file):
# 		keys = [key for key in h5_file.keys()]
# 		self.DIM         = len(h5_file[keys[1]].shape)
# 		self.DATA        = np.array(h5_file[keys[1]])
# 		self.DATA_UNITS  = h5_file[keys[1]].attrs['UNITS'][0].decode("utf-8")
# 		self.DATA_NAME   = h5_file[keys[1]].attrs['LONG_NAME'][0].decode("utf-8")
# 		self.TIME        = h5_file.attrs['TIME']
# 		self.DT          = h5_file.attrs['DT']
# 		self.TIME_UNITS  = h5_file.attrs['TIME UNITS'][0].decode("utf-8")


# 		self.AXIS1       = np.array(h5_file['AXIS']['AXIS1'])
# 		self.AXIS1_UNITS = h5_file[keys[0]]['AXIS1'].attrs['UNITS'][0].decode("utf-8")
# 		self.AXIS1_NAME  = h5_file[keys[0]]['AXIS1'].attrs['LONG_NAME'][0].decode("utf-8")
# 		self.NX          = h5_file[keys[1]].shape[self.DIM-1]
# 		self.DX          = (self.AXIS1[1]-self.AXIS1[0])/(self.NX)
# 		# self.X           = np.linspace(self.AXIS1[0], self.AXIS1[-1] - self.DX, num = self.NX)
# 		self.X           = np.linspace(self.AXIS1[0] + self.DX/2, self.AXIS1[-1] - self.DX/2, num = self.NX)
# 		# self.X           = self.AXIS1[0] + np.arange(self.NX) * self.DX

# 		if self.DIM > 1:
# 			self.AXIS2       = np.array(h5_file[keys[0]]['AXIS2'])
# 			self.AXIS2_UNITS = h5_file[keys[0]]['AXIS2'].attrs['UNITS'][0].decode("utf-8")
# 			self.AXIS2_NAME  = h5_file[keys[0]]['AXIS2'].attrs['LONG_NAME'][0].decode("utf-8")
# 			self.NY          = h5_file[keys[1]].shape[self.DIM-1 - 1]
# 			self.DY          = (self.AXIS2[1]-self.AXIS2[0])/(self.NY)
# 			self.Y   = np.linspace(self.AXIS2[0] + self.DY/2, self.AXIS2[-1] - self.DY/2, num = self.NY)
# 			# self.Y   = np.linspace(self.AXIS2[0], self.AXIS2[-1] - self.DY, num = self.NY)
# 			# self.Y           = self.AXIS2[0] + np.arange(self.NY) * self.DY

# 		if self.DIM == 3:
# 			self.AXIS3       = np.array(h5_file[keys[0]]['AXIS3'])
# 			self.AXIS3_UNITS = h5_file[keys[0]]['AXIS3'].attrs['UNITS'][0].decode("utf-8")
# 			self.AXIS3_NAME  = h5_file[keys[0]]['AXIS3'].attrs['LONG_NAME'][0].decode("utf-8")
# 			self.NZ          = h5_file[keys[1]].shape[self.DIM-1 - 2]
# 			self.DZ          = (self.AXIS3[1]-self.AXIS3[0])/(self.NZ)
# 			self.Z   = np.linspace(self.AXIS3[0] + self.DZ/2, self.AXIS3[-1] - self.DZ/2, num = self.NZ)
# 			# self.Z   = np.linspace(self.AXIS3[0], self.AXIS3[-1] - self.DZ, num = self.NZ)
# 			# self.Z           = self.AXIS3[0] + np.arange(self.NZ) * self.DZ

def get_quant_key(h5_keys):
    for key in h5_keys:
        if key != 'AXIS' and key != 'SIMULATION':
            return key

class vysxd_data_object(object):
    def __init__(self, h5_file):
        keys = [key for key in h5_file.keys()]
        
        quant_key = get_quant_key(keys)

        self.DIM         = len(h5_file[quant_key].shape)
        self.DATA        = np.array(h5_file[quant_key])
        # self.DATA_UNITS  = h5_file.attrs['UNITS'][0].decode("utf-8")
        self.DATA_NAME   = h5_file.attrs['NAME'][0].decode("utf-8")
        self.TIME        = h5_file.attrs['TIME']
        self.TIME_UNITS  = h5_file.attrs['TIME UNITS'][0].decode("utf-8")
        # self.DT          = h5_file[keys[1]].attrs['DT'][0]


        self.AXIS1       = np.array(h5_file[keys[0]]['AXIS1'])
        self.AXIS1_UNITS = h5_file['AXIS']['AXIS1'].attrs['UNITS'][0].decode("utf-8")
        self.AXIS1_NAME  = h5_file['AXIS']['AXIS1'].attrs['LONG_NAME'][0].decode("utf-8")
        self.NX          = h5_file[quant_key].shape[self.DIM-1]
        self.DX          = (self.AXIS1[1]-self.AXIS1[0])/(self.NX)
        self.X           = np.linspace(self.AXIS1[0], self.AXIS1[-1] - self.DX, num = self.NX)
        # self.X           = np.linspace(self.AXIS1[0] + self.DX/2, self.AXIS1[-1] - self.DX/2, num = self.NX)
        # self.X           = self.AXIS1[0] + np.arange(self.NX) * self.DX

        if self.DIM > 1:
            self.AXIS2       = np.array(h5_file['AXIS']['AXIS2'])
            self.AXIS2_UNITS = h5_file['AXIS']['AXIS2'].attrs['UNITS'][0].decode("utf-8")
            self.AXIS2_NAME  = h5_file['AXIS']['AXIS2'].attrs['LONG_NAME'][0].decode("utf-8")
            self.NY          = h5_file[quant_key].shape[self.DIM-1 - 1]
            self.DY          = (self.AXIS2[1]-self.AXIS2[0])/(self.NY)
            # self.Y   = np.linspace(self.AXIS2[0] + self.DY/2, self.AXIS2[-1] - self.DY/2, num = self.NY)
            self.Y   = np.linspace(self.AXIS2[0], self.AXIS2[-1] - self.DY, num = self.NY)
            # self.Y           = self.AXIS2[0] + np.arange(self.NY) * self.DY

        if self.DIM == 3:
            self.AXIS3       = np.array(h5_file['AXIS']['AXIS3'])
            self.AXIS3_UNITS = h5_file['AXIS']['AXIS3'].attrs['UNITS'][0].decode("utf-8")
            self.AXIS3_NAME  = h5_file['AXIS']['AXIS3'].attrs['LONG_NAME'][0].decode("utf-8")
            self.NZ          = h5_file[quant_key].shape[self.DIM-1 - 2]
            self.DZ          = (self.AXIS3[1]-self.AXIS3[0])/(self.NZ)
            # self.Z   = np.linspace(self.AXIS3[0] + self.DZ/2, self.AXIS3[-1] - self.DZ/2, num = self.NZ)
            self.Z   = np.linspace(self.AXIS3[0], self.AXIS3[-1] - self.DZ, num = self.NZ)
            # self.Z           = self.AXIS3[0] + np.arange(self.NZ) * self.DZ

class vysxd_raw_object(object):
	def __init__(self, h5_file):
		self.DIM         = len(h5_file.attrs['XMIN'])
		self.TIME        = h5_file.attrs['TIME'][0]
		self.TIME_UNITS  = h5_file.attrs['TIME UNITS'][0]
		self.NX          = h5_file.attrs['NX'][0]
		self.DT          = h5_file.attrs['DT'][0]
		self.DT          = h5_file.attrs['DT'][0]

		self.AXIS1       = np.array(h5_file[h5_file.keys()[0]]['AXIS1'])
		self.AXIS1_UNITS = h5_file[h5_file.keys()[0]]['AXIS1'].attrs['UNITS'][0]
		self.AXIS1_NAME  = h5_file[h5_file.keys()[0]]['AXIS1'].attrs['LONG_NAME'][0]
		self.NX          = h5_file[h5_file.keys()[1]].shape[self.DIM-1]
		self.DX          = (self.AXIS1[1]-self.AXIS1[0])/(self.NX-1)
		self.X   = np.linspace(self.AXIS1[0], self.AXIS1[1], num = self.NX)

		if self.DIM > 1:
			self.AXIS2       = np.array(h5_file[h5_file.keys()[0]]['AXIS2'])
			self.AXIS2_UNITS = h5_file[h5_file.keys()[0]]['AXIS2'].attrs['UNITS'][0]
			self.AXIS2_NAME  = h5_file[h5_file.keys()[0]]['AXIS2'].attrs['LONG_NAME'][0]
			self.NY          = h5_file[h5_file.keys()[1]].shape[self.DIM-1 - 1]
			self.DY          = (self.AXIS2[1]-self.AXIS2[0])/(self.NY-1)
			self.Y   = np.linspace(self.AXIS2[0], self.AXIS2[1], num = self.NY)

		if self.DIM == 3:
			self.AXIS3       = np.asarray(h5_file[h5_file.keys()[0]]['AXIS3'])
			self.AXIS3_UNITS = h5_file[h5_file.keys()[0]]['AXIS3'].attrs['UNITS'][0]
			self.AXIS3_NAME  = h5_file[h5_file.keys()[0]]['AXIS3'].attrs['LONG_NAME'][0]
			self.NZ          = h5_file[h5_file.keys()[1]].shape[self.DIM-1 - 2]
			self.DZ          = (self.AXIS3[1]-self.AXIS3[0])/(self.NZ-1)
			self.Z   = np.linspace(self.AXIS3[0], self.AXIS3[1], num = self.NZ)


def vysxd_get_data(filename):
	# Open first hdf5 file, save its contents to vys_data object, and close file
	file = h5py.File(filename, 'r')
	vysxd_data = vysxd_data_object(file)
	file.close()

	return vysxd_data
