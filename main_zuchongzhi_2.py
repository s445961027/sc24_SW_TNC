#-*- coding: utf-8 -*-
# @Author: guochu
# @Date:   2021-05-07 08:36:14
# @Last Modified by:   guochu
# @Last Modified time: 2021-05-15 22:14:00
import sys

# 获取当前 Python 解释器的路径
python_executable = sys.executable

# 获取当前 Python 解释器调用的库的路径列表
library_paths = sys.path

print("Python 解释器路径:", python_executable)
print("库的路径列表:")
for path in library_paths:
    print(path)


import quimb.tensor as qtn
import cotengra as ctg
from opt_einsum import contract_expression
from math import sqrt
import time
import json
import sys
from numpy import savez
from cotengra import ContractionTree
from check_pos import prepare_compute_amplitudes

def prod(it):
	"""Compute the product of sequence of numbers ``it``.
	"""
	x = 1
	for i in it:
		x *= i
	return x
	
def dynary(x, bases):
	"""Represent the integer ``x`` with respect to the 'dynamical' ``bases``.
	Gives a way to reliably enumerate and 'de-enumerate' the combination of
	all different index values.

	Examples
	--------

		>>> dynary(9, [2, 2, 2, 2])  # binary
		[1, 0, 0, 1]

		>>> dynary(123, [10, 10, 10])  # decimal
		[1, 2, 3]

		>>> # arbitrary
		>>> bases = [2, 5, 7, 3, 8, 7, 20, 4]
		>>> for i in range(301742, 301752):
		...     print(dynary(i, bases))
		[0, 3, 1, 1, 2, 5, 15, 2]
		[0, 3, 1, 1, 2, 5, 15, 3]
		[0, 3, 1, 1, 2, 5, 16, 0]
		[0, 3, 1, 1, 2, 5, 16, 1]
		[0, 3, 1, 1, 2, 5, 16, 2]
		[0, 3, 1, 1, 2, 5, 16, 3]
		[0, 3, 1, 1, 2, 5, 17, 0]
		[0, 3, 1, 1, 2, 5, 17, 1]
		[0, 3, 1, 1, 2, 5, 17, 2]
		[0, 3, 1, 1, 2, 5, 17, 3]

	"""
	bs_szs = [prod(bases[i + 1:]) for i in range(len(bases))]
	dx = []
	for b in bs_szs:
		div = x // b
		dx.append(div)
		x -= div * b
	return dx


def get_sliced_arrays(arrays, sliced_indexes, sliced_sizes, changing_indexes, inputs, i):
	"""Generate the tuple of array inputs corresponding to slice ``i``.
	"""
	temp_arrays = list(arrays)

	# e.g. {'a': 2, 'd': 7, 'z': 0}
	locations = dict(zip(sliced_indexes, dynary(i, sliced_sizes)))

	for c in changing_indexes:
		# the indexing object, e.g. [:, :, 7, :, 2, :, :, 0]
		selector = tuple(locations.get(ix, slice(None)) for ix in inputs[c])
		# re-insert the sliced array
		temp_arrays[c] = temp_arrays[c][selector]

	return temp_arrays

def compute_marginal(data):
	inputs = data['inputs']
	eq = data['eq']
	eq_sliced = data['eq_sliced']
	sliced_indexes = data['sliced_indexes']
	size_dict = data['size_dict']
	arrays = data['arrays']	
	constant_indexes = data['constant_indexes']
	changing_indexes = data['changing_indexes']
	sliced_sizes = data['sliced_sizes']
	sliced_path = data['sliced_path']
	nslices = data['nslices']
	shapes_sliced = data['shapes_sliced']
	fixed_qubits = data['fixed_qubits']

	# sc = ctg.SlicedContractor(eq, arrays, sliced_indexes, optimize=get_optimizer(2**27, 120), size_dict=size_dict)
	# p_marginal = abs(sc.contract_all(backend='jax'))
	contract_expr = contract_expression(eq_sliced, *shapes_sliced, optimize=sliced_path)

	backend = 'jax'

	temp_arrays = get_sliced_arrays(arrays, sliced_indexes, sliced_sizes, changing_indexes, inputs, 0)
	p_marginal = contract_expr(*temp_arrays, backend=backend)
	for i in range(1, nslices):
		temp_arrays = get_sliced_arrays(arrays, sliced_indexes, sliced_sizes, changing_indexes, inputs, i)
		p_marginal = p_marginal + contract_expr(*temp_arrays, backend=backend)
	nfact = 2**len(fixed_qubits)
	return abs(p_marginal)**2 / nfact

def compute_amplitudes(data):
	inputs = data['inputs']
	eq = data['eq']
	eq_sliced = data['eq_sliced']
	sliced_indexes = data['sliced_indexes']
	size_dict = data['size_dict']
	# arrays = data['arrays']	
	constant_indexes = data['constant_indexes']
	changing_indexes = data['changing_indexes']
	sliced_sizes = data['sliced_sizes']
	sliced_path = data['sliced_path']
	nslices = data['nslices']
	shapes_sliced = data['shapes_sliced']

	all_arrays = data['all_arrays']
	backend = 'jax'

	amps = []
	for arrays in all_arrays:
		contract_expr = contract_expression(eq_sliced, *shapes_sliced, optimize=sliced_path)
		temp_arrays = get_sliced_arrays(arrays, sliced_indexes, sliced_sizes, changing_indexes, inputs, 0)
		amp = contract_expr(*temp_arrays, backend=backend)
		for i in range(1, nslices):
			temp_arrays = get_sliced_arrays(arrays, sliced_indexes, sliced_sizes, changing_indexes, inputs, i)
			amp = amp + contract_expr(*temp_arrays, backend=backend)
		amps.append(abs(amp)**2)

	return amps



def load_circuit(n=53, depth=10, seed=0, elided=0, sequence='ABCDCDAB', swap_trick=False):
	file = f'circuit_n{n}_m{depth}_s{seed}_e{elided}_p{sequence}.qsim'

	if swap_trick:
		gate_opts={'contract': 'swap-split-gate', 'max_bond': 2}  
	else:
		gate_opts={}
	
	# instantiate the `Circuit` object that 
	# constructs the initial tensor network:
	return qtn.Circuit.from_qasm_file(file, gate_opts=gate_opts)


def parse_line(circuit, qubits_map, line):
	item = line.split()
	gate_round = int(item[0])
	gate = item[1]
	if gate == 'x_1_2':
		pos = int(item[2])
		circuit.apply_gate('X_1_2', qubits_map[pos], gate_round=gate_round)
	elif gate == 'y_1_2':
		pos = int(item[2])
		circuit.apply_gate('Y_1_2', qubits_map[pos], gate_round=gate_round)
	elif gate == 'hz_1_2':
		pos = int(item[2])
		circuit.apply_gate('HZ_1_2', qubits_map[pos], gate_round=gate_round)
	elif gate == 'rz':
		pos = int(item[2])
		theta = float(item[3])
		circuit.apply_gate('RZ', theta, qubits_map[pos], gate_round=gate_round)
	elif gate[:8] == 'fsimplus':
		pos1 = int(item[6])
		pos2 = int(item[7])
		theta = float(gate[9:-1])
		phi = float(item[2][:-1])
		deltap = float(item[3][:-1])
		deltam = float(item[4][:-1])
		deltamoff = float(item[5][:-1])
		# print(pos1, pos2, theta, phi, deltap, deltam, deltamoff)
		circuit.apply_gate('FSIMG', theta, -deltam, -deltamoff, -deltap, phi, qubits_map[pos1], qubits_map[pos2])
	else:
		raise ValueError('unknown gate %s'%gate)

def parse_auxiliary(path_name):
	#print('read auxiliary information from path %s'%path_name)
	with open(path_name, 'r') as f:
		data = f.read()
		data = json.loads(data)

	# print(data['map'])
	index_mapping = {a:b-1 for a, b in data["map"]}
	return index_mapping

def parse_circuit_from_file(circuit_path, qubits_map):
	lines = open(circuit_path).read().split('\n')
	n = int(lines[0])
	circuit = qtn.Circuit(n)
	for j in range(1,len(lines)):
		parse_line(circuit, qubits_map, lines[j])
	return n, circuit


def parse_all_information(circuit_dir):
	auxiliary_dir = circuit_dir + '.json'
	qubits_map = parse_auxiliary(auxiliary_dir)
	n, circuit = parse_circuit_from_file(circuit_dir, qubits_map)
	return n, circuit, qubits_map

def parse_basis(basis_dir, number=0):
	lines = open(basis_dir).read().split('\n')
	if number <= 0:
		number = len(lines)
	#print('number of amplitudes to be measured %s'%number)
	return lines[:number]

def read_1(data_file):
    arr = []
    with open(data_file) as fin:
        for line in fin:
            words = re.split(r' ',line)
            #print(words[0])
            arr.append(words[0])
        #print(arr)
    return(arr)

def read_2(data_file):
    arr = []
    with open(data_file) as fin:
        for line in fin:
            line = line.strip()
            #words = re.split(r' ',line)
            #print(words[0])
            #arr.append(words[0])
            arr.append(line)
        #print(arr)
    return(arr)

def load_circuit(n=53, depth=20, seed=0, elided=0, sequence='ABCDCDAB', swap_trick=False, path =''):
	file =  mpath+f'circuit_n{n}_m{depth}_s{seed}_e{elided}_p{sequence}.qsim'

	if swap_trick:
		gate_opts={'contract': 'swap-split-gate', 'max_bond': 2}  
	else:
		gate_opts={}
	
	# instantiate the `Circuit` object that 
	# constructs the initial tensor network:
	#return qtn.Circuit.from_qasm_file(file, gate_opts=gate_opts)
	return qtn.circuit.Circuit.from_qasm_file(file,gate_opts=gate_opts)

if __name__ == '__main__':
	
	sys.setrecursionlimit(100000)
	#num_args = len(sys.argv)
	#iter = -1
	mpath = sys.argv[1]
	zcz_depth = sys.argv[2]
	qubits_num = 60
	#qbits_num = int(sys.argv[3])
	circuit_dir = mpath + '/circuit.txt'
	n, circuit, qubits_map = parse_all_information(circuit_dir)
	basis = '0' * qbits_num
 
	prepare_compute_amplitudes(iter,mpath,circuit, basis, max_size=31, max_search_time=720)



