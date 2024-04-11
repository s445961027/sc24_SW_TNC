import os
import sys
import quimb.tensor as qtn
import cotengra as ctg
import numpy as np
from cotengra.plot import tree_to_networkx
import networkx as nx
import time
import copy
import random
import sys
import math
from tran import Trans
from choose_slice import choose_slice_index,choose_slice_index_szq_v2,check_slice_index,path_to_eq
from choose_slice import slice_refine,path_to_node
from debug import find_trunks




def get_optimizer_original(target_size, max_search_time):
	
	opt = ctg.HyperOptimizer(
		methods=['kahypar'],
		max_time=max_search_time,              # just search for 2 minutes
		max_repeats=100,
		progbar=True,
		parallel=True,
		#parallel='ray',
		#parallel=False,
		minimize='flops', #{'size','flops','combo'},what to target
		#slicing_reconf_opts={'target_size': target_size},
		#slicing_opts={'target_size': target_size,'inplace':True},
	)
	return opt



def get_optimizer_sliced(target_size, max_search_time):
	
	opt = ctg.HyperOptimizer(
		methods=['kahypar'],
		max_time=max_search_time,              # just search for 2 minutes
		max_repeats=100,
		progbar=True,
		parallel=True,
		#parallel='ray',
		#parallel=False,
		minimize='flops', #{'size','flops','combo'},what to target
		slicing_reconf_opts={'target_size': target_size},
		#slicing_opts={'target_size': target_size,'inplace':True},
	)
	return opt


def get_amplitude_tn(circ, dtype, basis=None):
	basis = '0' * circ.N
	sampler = qtn.MPS_computational_state(basis)
	tn = circ.psi & sampler
	tn.full_simplify_(output_inds=[])
	tn.astype_(dtype)
	return tn

def path_str_to_int(path):
	tran = Trans()
	steps = len(path)
	k = 1 
	path_int = list()
	for step in path:
		step_new = [int(step[0]), int(step[1]), steps - k]
		path_int.append(step_new)
		k += 1
	print("path int is",path_int)
	newPath = trans_path_index(path_int)
	return newPath

def trans_path_index(path):
		steps = len(path)
		align = [0]*(steps+1)
		simu = list()
		for i in range(steps+1):
			simu.append(i)
		newPath = list()
		k = 1
		for step in path:
			step_new = copy.deepcopy(step)
			step_new[0] = step[0] + align[step[0]]
			step_new[1] = step[1] + align[step[1]]	
			step_new[2] = steps + k
			newPath.append(step_new)
			del simu[step[1]]
			del simu[step[0]]
			simu.append(steps+k)
			for i in range(len(simu)):
				align[i] = simu[i] - i
			k += 1
		return newPath

def load_path(filename):
	with open(filename) as f:
		data = f.readline()
		data_tensor = list()
		path_str = data.strip("path:")
		path = path_str.split("), (")
		path[0] = path[0].strip(" ((")
		path[len(path) - 1] = path[len(path) - 1].strip("))\n")
		steps = len(path)
		for i in range(steps):
			tensor = f.readline().split(" ")
			tensor_u = list()
			for ele in tensor:
				ele = ele.strip("\n")
				tensor_u.append(ele)
#				print(ele.decode("UTF-8"))
#			print(len(tensor_u[1]))
			data_tensor.append(tensor_u)

	k = 1
	path_int = list()
	for step in path:
		step = step.split(", ")
		step_new = [int(step[0]), int(step[1]), steps - k]
		path_int.append(step_new)
		k += 1
	return path_int, data_tensor


def write2slicetxt_notrans(mpath,tree_s,info,original_time,newPath,iter):

	#original_time  = tree.contraction_cost()
	slice_time = tree_s.contraction_cost()

	filename = mpath+f'/trunk_info/sliced_{iter}.txt'
	file = open(filename,'w')
	input_subscripts = info.input_subscripts
	
	
	input_list = input_subscripts.split(',')
	#导出未slice前的tensor的符号字符串
	for item in input_subscripts:
		file.write(item)
	file.write('\n')
	

	#导出要slice的边的符号
	slice_list = tree_s.sliced_inds
	slice_tensor_id = []
	
	#找出被slice的tensor的id
	
	for slice_index in slice_list:
		for i,item in enumerate(input_list):
			if slice_index in item:
				slice_tensor_id.append(i)
	slice_tensor_id = list(set(slice_tensor_id))
	slice_tensor_id.sort()
	
	#导出被slice的tensor的id
	
	file.write(str(slice_tensor_id))
	file.write('\n')
	file.write(str(slice_list))
	file.write('\n')

	#导出slice后的tensor的符号字符串
	indices_after_slice=[]
	
	for string in input_list:
		for char in slice_list:
			if char in string:
				string = string.replace(char,'')
		indices_after_slice.append(string)

	for item in indices_after_slice:
		file.write(item)
		file.write(',')
	file.write('\n')
	#ind_after_slice = str(indices_after_slice).replace("'","").replace("[","").replace("]","").replace(" ","")
	#ind_after_slice = ind_after_slice.encode().decode("unicode-escape")
	#print(ind_after_slice)
	#file.write(ind_after_slice)
	#file.write('\n')
	
	# file.write(str(tree.get_path()))
	# file.write('\n')
	#path = tree_s.get_path()
	file.write(str(newPath))
	file.write('\n')
	file.write("original time cost is :"+str(original_time))
	file.write('\n')
	file.write("slice time cost is :"+str(slice_time))
	file.write('\n')
	file.write(str(get_contraction_from_tree(tree_s)))
	file.close()
	

	return input_list, slice_list, slice_tensor_id, indices_after_slice
def write2slicetxt_trans(mpath,tree_s,info,original_time,newPath):

	#original_time  = tree.contraction_cost()
	slice_time = tree_s.contraction_cost()

	filename = mpath+'/sliced_trans.txt'
	file = open(filename,'w')
	input_subscripts = info.input_subscripts
	
	input_list = input_subscripts.split(',')
	print(input_list)
	#input_list = input_subscripts
	#导出未slice前的tensor的符号字符串
	for item in input_subscripts:
		file.write(item)
	file.write('\n')
	

	#导出要slice的边的符号
	slice_list = tree_s.sliced_inds
	slice_tensor_id = []
	
	#找出被slice的tensor的id
	
	for slice_index in slice_list:
		for i,item in enumerate(input_list):
			if slice_index in item:
				slice_tensor_id.append(i)
	slice_tensor_id = list(set(slice_tensor_id))
	slice_tensor_id.sort()
	
	#导出被slice的tensor的id
	
	file.write(str(slice_tensor_id))
	file.write('\n')
	file.write(str(slice_list))
	file.write('\n')

	#导出slice后的tensor的符号字符串
	indices_after_slice=[]
	# for item in slice_list:
	# 	tree_s.remove_ind(item,inplace=False)
	# mymap = tree_s.gen_leaves()
	# for item in mymap:
	# 	indices_after_slice.append(tree_s.get_inds(item))
	for string in input_list:
		for char in slice_list:
			if char in string:
				string = string.replace(char,'')
		indices_after_slice.append(string)

	for item in indices_after_slice:
		file.write(item)
		file.write(',')
	file.write('\n')
	
	file.write(str(newPath))
	file.write('\n')
	file.write("original time cost is :"+str(original_time))
	file.write('\n')
	file.write("slice time cost is :"+str(slice_time))
	file.write('\n')

	file.write(str(get_contraction_from_tree(tree_s)))
	file.close()
	

	return input_list, slice_list, slice_tensor_id, indices_after_slice

#导出tensor
def write2array(mpath,tn,iter):
	filename = mpath+f'/array/array.txt'
	
	arrays = [t.data.tolist() for t in tn]
	
	file = open(filename,'w')
	for array in arrays:

		file.write(str(array) + '\n')
	file.close()
	

class Slice_Cost:
	def __init__(self,tree_or_info,slice_inds):
		self.tree=tree_or_info
		self.cost0=ctg.slicer.ContractionCosts.from_contraction_tree(tree_or_info)
		self.costs={frozenset(): self.cost0}#这个costs记录每一步slice带来的overhead
		self.slice_inds=slice_inds
	
	def cal_cost(self):
		ix_sl = frozenset()
		cost = self.costs[ix_sl]
		for ix in self.slice_inds:
			next_ix_sl = ix_sl | frozenset([ix])
			next_cost = self.costs[next_ix_sl] = cost.remove(ix)
			
			ix_sl = next_ix_sl
			cost = next_cost
		#返回最后的cost
		return cost

#获取每一步收缩 的维度信息
def get_contraction_from_tree(tree):
	every_step=[]
	for i, (p, l, r) in enumerate(tree.traverse()):
		p_legs, l_legs, r_legs = map(tree.get_legs, [p, l, r])
		p_inds, l_inds, r_inds = map(tree.get_inds, [p, l, r])
		# print sizes and flops
		p_flops = tree.get_flops(p)
		p_sz, l_sz, r_sz = (math.log2(tree.get_size(node)) for node in [p, l, r])
		if l_sz >= 13 and r_sz >= 13:
			every_step.append([i,l_sz,r_sz,p_sz])
	return every_step



def find_optimized_path_info(basis,mpath,circ, target_size, max_search_time,iter):
	#original_opt = get_optimizer_original(2**target_size, max_search_time)
	opt = get_optimizer_sliced(2**target_size, max_search_time)
	#basis = '0'*56
	print(basis)
	#开openleg

	# # 垃圾方法，不太行
	
	#opt = ctg.HyperOptimizer(reconf_opts={},parallel=True,progbar=True)
	rehs = circ.amplitude_rehearse(b=basis,optimize=opt)
	tn = rehs['tn']
	#print(rehs['tn'].data)
	info = rehs['info']
	#print(rehs['info'].input_subscripts)
	#print(tn.arrays)
	# rehs = circ.sample_chaotic_rehearse(circ.calc_qubit_ordering()[-circ.N:], optimize=opt)
	# print(rehs)
	#sys.exit()
	#tn = get_amplitude_tn(circ, 'complex64')
	
	#info = tn.contract(all, optimize=opt, get='path-info', output_inds=[])

	#info = tn.contraction_info(optimize=opt)
	print(info.input_subscripts)
	#print(tn.arrays)
	# for key in rehs.keys():
	# 	tensor_net = rehs[key]['tn']
	# # 	info = rehs[key]['info']
	# # 	#write2array(mpath,tensor_net)
	# info = tensor_net.contract(all, get = 'path-info')
	# inputs,outputs,size_dict = tensor_net.get_inputs_output_size_dict()
	# print("get_equation is :" ,tensor_net.get_equation())
	# #print(inputs)
	# print(tensor_net.arrays)
	# print(tensor_net.ind_map)
	# symbol_map = tensor_net.get_symbol_map()
	# ind_map = tensor_net.ind_map
	# for item in tensor_net.outer_inds():
	# 	print(symbol_map[item])
	# print(outputs)
	#print(tensor_net.outer_inds())
	#tensor_net = rehs['tn']
	#tensor_net.full_simplify_(output_inds=[])
	#tensor_net.astype_('complex64')
	
	#write2array(mpath,tn)	
	#print(rehs['tnn'])	
	
	#write2array(mpath,tn,iter)
	
	path = info.path
	
	#debug 
	# filename = mpath+'/path.txt'
	# file = open(filename,'r')
	# file.write(str(path))
	# file.close()

	#node = path_to_node(path)

	#print("node is :",node)
	tree_s = opt.get_tree()
	trees_path = tree_s.get_path()
	print("trees_path is:",trees_path)
	print("inputs is " ,tree_s.inputs)
	print("outputs is ",tree_s.output)
	slice_list = tree_s.sliced_inds
	print("ctg 得到的slice list:",slice_list)

	#利用path建一棵没有slice的完整的收缩树
	tree_original = ctg.ContractionTree.from_path(inputs = tree_s.inputs,output=tree_s.output,size_dict=tree_s.size_dict,path=trees_path)
	#tree_tmp = ctg.ContractionTree.from_path(inputs = tree.inputs,output=tree.output,size_dict=tree.size_dict,path=trees_path)


	#实验一下官方提供的两种官方提供的计算时间复杂度的方法
	print("通过path重建得到的tree的total_flops:",tree_original.total_flops)#
	print("通过path重建得到的tree的contraction_cost:",2 * tree_original.contraction_cost())#乘加
	contraction_cost = ctg.slicer.ContractionCosts.from_contraction_tree(tree_original)#乘+加
	print("ctg的slicer 通过path重建得到的tree的contraction_cost:",contraction_cost.total_flops)#乘+加
	ctg_slice_cost = Slice_Cost(tree_original, list(slice_list))
	print("ctg得到的slice list的overhead: ", ctg_slice_cost.cal_cost())
	print(tree_s.contraction_cost())
	_,_,_,lhs_sliced = write2slicetxt_notrans(mpath, tree_s, info, tree_original.contraction_cost(),trees_path,iter)

	newPath = path_str_to_int(trees_path)
	tensor_info = path_to_eq(info,newPath)
	#先用自己的方法选出slice_set
	#lf_sl = choose_slice_index(info, node, target_size)
	#slice_set = list(lf_sl)

	#检测cotengra方法的合法性
	Error_loc0 = check_slice_index(tensor_info,slice_list,target_size)
	print("ctg的error:",Error_loc0)
	#非线性压缩成线性，线性trunk保存在Stems中
	szqPath,Stems,trunk_num = find_trunks(tensor_info,newPath,target_size)
	lf_sl = choose_slice_index_szq_v2(target_size,tensor_info,Stems)
	slice_set = list(lf_sl)
	Error_loc1 = check_slice_index(tensor_info,slice_set,target_size)
	slice_cost = Slice_Cost(tree_original,slice_set)
	slice_cal_cost =  slice_cost.cal_cost()
	print("refine前的overhead: ",slice_cal_cost)
	print("refine前的error:",Error_loc1)
	#再对slice_set refine
	refine_slice_set = slice_refine(szqPath, slice_set, info, target_size)
	Error_loc2 = check_slice_index(tensor_info,refine_slice_set,target_size)
	#利用写好的slice_cost类来计算 裁掉特定的index带来的overhead
	refine_slice_cost = Slice_Cost(tree_original,refine_slice_set)
	#print("每一个slice_inds带来的overhead: ",slice_cost.costs)
	refine_slice_cal_cost = refine_slice_cost.cal_cost()
	print("refine后的overhead: ",refine_slice_cal_cost)
	print("refine后的error:",Error_loc2)

	filename = mpath+f'/trunk_info/sliced_{iter}.txt'
	file = open(filename,'a')
	file.write(str(szqPath))
	file.write('\n')
	
	file.write(str(trunk_num))
	file.write('\n')
	file.write(str(Error_loc0))
	file.write('\n')
	file.write(str(Error_loc1))
	file.write('\n')
	file.write(str(Error_loc2))
	file.write('\n')
	file.write('slice_set is :\n{}'.format(slice_set))
	file.write('\n')
	file.write('refine_slice_set is :\n{}'.format(refine_slice_set))
	file.write('\n')
	file.write('ctg生成的slice list 的overhead : {}'.format(ctg_slice_cost.cal_cost()))
	file.write('\n')
	file.write('slice_set 的overhead: {}'.format(slice_cal_cost))
	file.write('\n')
	file.write('refine_slice_set 的overhead: {}'.format(refine_slice_cal_cost))
	file.write('\n')
	file.write('ctg_slice 的数量 : {}'.format(len(slice_list)))
	file.write('\n')
	file.write('slice_refine 的数量: {}'.format(len(refine_slice_set)))
	file.close()

	#_,_,_,_ = write2slicetxt_notrans(mpath, tree_s, info, tree_original.contraction_cost(),trees_path,iter)
	
	#_,_,_,_ = write2slicetxt_trans(mpath, tree_s, info, tree_original.contraction_cost(),newPath)
	
	return info,opt
	#return rehs, original_opt,sliced_opt

def find_path_info(circ, path):
	tn = get_amplitude_tn(circ, 'complex64')

	info = tn.contract(all, optimize=path, get='path-info', output_inds=[])
	
	return info

def get_last_line(filename):
	try:
		filesize=os.path.getsize(filename)
		if filesize == 0:
			return None
		else:
			with open(filename,'rb') as fp:
				offset = -8
				while -offset < filesize:
					fp.seek(offset,2)
					lines=fp.readlines()
					if len(lines) >= 2:
						return lines[-1].decode()
					else:
						offset *= 2
				fp.seek(0)
				lines=fp.readlines()
				return lines[-1].decode()
	except FileNotFoundError:
		print(filename + ' not found! ')
		return None


def prepare_compute_marginal(circuit, n_marginal, max_size, max_search_time):
	target_size = 2**max_size
	marginal_qubits = circuit.calc_qubit_ordering()[-n_marginal:]
	print('marginal_qubits:',marginal_qubits)
	#rest_qubits = circuit.calc_qubit_ordering()[:(circuit.N - n_marginal)]
	opt = get_optimizer_original(target_size, max_search_time)

	backend = 'jax'
	#fix = {k:np.random.choice(('0', '1')) for k in rest_qubits}

	info = circuit.compute_marginal(where=marginal_qubits, optimize=opt, backend=backend, target_size=target_size, rehearse=True)
	tn = get_amplitude_tn(circuit, 'complex64')
	mpath = "./"
	write2array(mpath,tn)
	for key in info.keys():
		print(key)
	print(type(info[tn]))
	return info


def prepare_compute_amplitudes(iter,mpath,circuit,basis, max_size, max_search_time):

	#info = prepare_compute_marginal(circuit, 0, max_size, max_search_time)
	info, sliced_opt = find_optimized_path_info(basis,mpath,circuit, max_size, max_search_time,iter)
	#print(rehs)
	#path = opt.path
	# tree = opt.get_tree()
	# path = tree.get_path()
	#print(len(tree.sliced_inds),tree.sliced_inds)
	#print(path)
	
	
