# -*- coding: utf-8 -*-

import torch

use_cuda = False
device_num = 0
_zeros = None
_ones = None
_tensor_type = None
_long_tensor_type = None
_cuda = None

def cpuzeros(*size, device=None):
	return torch.FloatTensor(*size).fill_(0)
def cpuones(*size, device=None):
	return torch.FloatTensor(*size).fill_(1)
def cpucuda(x, device=None):
	return x

def getdevice(device):
	if device is None:
		return device_num
	elif isinstance(device, torch.Tensor):
		return device.get_device()
	else:
		return device
def gpuzeros(*size, device=None):
	device = getdevice(device)
	with torch.cuda.device(device):
		return torch.cuda.FloatTensor(*size).fill_(0)
def gpuones(*size, device=None):
	device = getdevice(device)
	with torch.cuda.device(device):
		return torch.cuda.FloatTensor(*size).fill_(1)
def gpucuda(x, device=None):
	device = getdevice(device)
	return x.cuda(device)

# pylint: disable=W0603
def init(device=None, __cuda=False):
	global _cuda, _zeros, _ones, use_cuda, device_num, _tensor_type, _long_tensor_type
	if __cuda:
		use_cuda = True
		device_num = device
		_cuda = gpucuda
		_zeros = gpuzeros
		_ones = gpuones
		_tensor_type = torch.cuda.FloatTensor
		_long_tensor_type = torch.cuda.LongTensor
	else:
		_cuda = cpucuda
		_zeros = cpuzeros
		_ones = cpuones
		_tensor_type = torch.FloatTensor
		_long_tensor_type = torch.LongTensor

def Tensor(*x, device=None):
	global use_cuda
	if not use_cuda:
		return _tensor_type(*x)
	else:
		device = getdevice(device)
		with torch.cuda.device(device):
			return _tensor_type(*x)

def LongTensor(*x, device=None):
	global use_cuda
	if not use_cuda:
		return _long_tensor_type(*x)
	else:
		device = getdevice(device)
		with torch.cuda.device(device):
			return _long_tensor_type(*x)

def cuda(*x, device=None):
	return _cuda(*x, device=device)

def zeros(*size, device=None):
	return _zeros(*size, device=device)

def ones(*size, device=None):
	return _ones(*size, device=device)
