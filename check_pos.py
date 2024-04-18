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
from trans import Trans as T
from tran import Trans

def find_stems(path,indices,target_dim):
	signal=[]
	stems=[]
	stem = []
	for step in path:
		if step not in signal:
			dfs(step,stem,stems,path,indices,target_dim,signal)
			stem = []
	return stems

def dfs(step,stem,stems,path,indices,target_dim,signal):
	#若不满足需求，则直接将当前stem加入stems
	if len(step) == 0:
		if len(stem) > 0:
			stems.append(stem)
		stem = []
		return

	if len(indices[step[0]]) < 13 and len(indices[step[1]]) < 13 :
		#signal.append(step)
		if len(stem) > 0:
			stems.append(stem)
		stem = []
		return	
	
	if len(indices[step[0]]) > 13 and len(indices[step[1]]) > 13 :
		#signal.append(step)
		if len(stem) > 0:
			stems.append(stem)
		stem = []
		return	

	if step in signal:
		if len(stem) > 0:
			stems.append(stem)
		stem = []
		return	
	
	signal.append(step)
	stem.append(step)

	#找到step[2]出现的位置
	next_step=[]
	#寻找当前step的输出张量作为输入大张量的那个step作为下一个step
	for next_contract in path:
		if step[-1] == next_contract[1]:
			next_step = next_contract
			break
	
	dfs(next_step,stem,stems,path,indices,target_dim,signal)



#将大的张量放在三元组的后面，小的张量放在前面
def exchange_A_B_szq(path,indices):
	newpath =[]
	for step in path:
		if len(indices[step[0]]) > len(indices[step[1]]):
			tmp = [ step[1],step[0],step[2] ]
			newpath.append(tmp)
		else:
			newpath.append(step)
	return newpath


def get_newpath_szq(newpath,stem):
	path = []
	path_tail=[]
	for step in newpath:
		if step not in stem and step[2] < stem[-1][2]:
			path.append(step)
		elif step not in stem and step[2] > stem[-1][2]:
			path_tail.append(step)
	path_length = len(path)

	for step in stem:
		path.append(step)
	
	for step in path_tail:
		path.append(step)
		
	return path,path_length


def dfs_search(tensor_a, stem, newPath, stem_tensor,end_tensor):
	tensor_b = -1
	#在newPath中找到以tensor_a作为输入的path
	as_input_index = -1
	for as_input_index,step in enumerate(newPath):
		if tensor_a in step[:2]:
			if step.index(tensor_a) == 0:
				tensor_b = step[1]
			else:
				tensor_b = step[0]
			break
	#若张量a或者张量b已经在stem_tensor中了，就直接返回
	if tensor_a in stem_tensor or tensor_b in stem_tensor or tensor_a == end_tensor:
		return
	stem_path = newPath[as_input_index]
	as_output_index = -1
	#找到tensor a作为输出tensor的stem中的下标
	for as_output_index,step in enumerate(stem):
		if tensor_a == step[2]:
			break
	stem.insert(as_output_index+1,stem_path)
	# stem_tensor.append(stem_path[0])
	# stem_tensor.append(stem_path[1])

	new_output_tensor = stem_path[2]
	dfs_search(new_output_tensor,stem,newPath,stem_tensor,end_tensor)	





#树链剖分
#输入：树的节点列表，节点的dict[父节点:左孩子，右孩子],用于对应树的node 
#stems为所有的stem的列表,
#node为当前节点，利用path中的[1,2,3]的张量序号就可以代表一个节点
#还需要考虑一个问题：要做融合段，还要确保每一步生成的张量C在下一个收缩步中是大张量，而不是小张量
def dfs_search_stem(node_dict,stems,node,indices,trunk_num):
	
	#判断该节点是在树中的收缩节点上，即非叶子节点上，若不是收缩生成的张量，则直接返回空列表
	if node not in node_dict:
		return []
	
	left_node = node_dict[node][0]
	right_node = node_dict[node][1]
	tmp_node = [left_node, right_node, node]

	#递归
	left_stem = dfs_search_stem(node_dict,stems,left_node,indices,trunk_num)
	right_stem = dfs_search_stem(node_dict,stems,right_node,indices,trunk_num)
	#判断该节点的A张量维度和B张量维度是否都大于13，若是，则当前stem直接返回，不再续了
	if len(indices[left_node]) >= 13 and len(indices[right_node]) >= 13:
		if len(left_stem) > 0:
			stems.append(left_stem)
		if len(right_stem) > 0:
			stems.append(right_stem)
		trunk_num = trunk_num + 1
		#这里可以添加统计trunk
		stems.append([tmp_node])
		#print("big path is",tmp_node)
		return []
	#判断该节点的A张量和B张量的维度是否都小于13，若是，也把这个stem返回，不续了
	elif len(indices[left_node]) < 13 and len(indices[right_node]) < 13:
		if len(left_stem) > 0:
			stems.append(left_stem)
		if len(right_stem) > 0:
			stems.append(right_stem)
		stems.append([tmp_node])
		#print("big path is",tmp_node)
		return []
	#若A B 张量一大一小，即一个大于13 一个小于13，则该把当前收缩路径加入到左右哪个stem中呢？
	#规则： 1.若左边的stem长度不为0，且左边stem生成的新张量为当前节点的输入大张量
	#      2.若右边的stem长度不为0，且右边stem生成的新张量为当前节点的输入大张量
	#      3.若左边的stem长度为0，且右边的stem长度不为0，但右边的stem生成的新张量不为当前节点的输入大张量
	#      4.若右边的stem长度为0，且左边的stem长度不为0，但左边的stem生成的新张量不为当前节点的输入大张量
	#      5.若左右两边的stem都为0,则直接返回包含当前节点的一个[[A,B,C]]
	else:

		if len(left_stem) > 0 and left_stem[-1][-1] == right_node:
			left_stem.append(tmp_node)
			if len(right_stem) > 0:
				stems.append(right_stem)
			return left_stem
		elif len(right_stem) > 0 and right_stem[-1][-1] == right_node:
			right_stem.append(tmp_node)
			if len(left_stem) > 0:
				stems.append(left_stem)
			return right_stem

		elif len(left_stem) == 0 and len(right_stem) > 0 and right_stem[-1][-1] != right_node:
			left_stem.append(tmp_node)
			stems.append(right_stem)
			return left_stem
		
		elif len(right_stem) == 0 and len(left_stem) > 0 and left_stem[-1][-1] != right_node:
			right_stem.append(tmp_node)
			stems.append(left_stem)
			return right_stem

		elif len(left_stem) == 0 and len(right_stem) == 0:
			left_stem.append(tmp_node)
			return left_stem


#带branch，然后利用树链剖分，将多个连续的stem存入stems返回
def detect_stems_szq_v2(newPath,indices,target_dim,trunk_num):
	stem = copy.deepcopy(newPath) 
	#保留所有包含大于13维的张量的收缩点
	stem_output = []
	for step in newPath:
		if len(indices[step[0]]) < target_dim and len(indices[step[1]]) < target_dim:
			stem.remove(step)
	
	end_tensor = stem[-1][-1]
	#把stem中所有分散的节点连接起来了
	stem_tensor = []
	for path in stem:
		stem_tensor.append(path[0])
		stem_tensor.append(path[1])
		if len(indices[path[2]]) < target_dim:
			stem_output.append(path[2])
	#stem_out用于存储容易出问题的张量的id
	#然后再遍历stem_output,查看与该tensor做收缩的张量是否也是小于14维的,若大于14维说明这个张量没问题，将其从stem_out中删除
	for tensor_a in stem_output:
		tensor_b = 0
		for step in newPath:
			#只检查前两个元素
			if tensor_a in step[:2]:
				tensor_a_index = step.index(tensor_a)
				if tensor_a_index == 0:
					tensor_b = step[1]
				else:
					tensor_b = step[0]
				#若与做a收缩的张量b大于等于14维，则将a从stem_out中移除
				if len(indices[tensor_b]) >= target_dim:
					stem_output.remove(tensor_a)


	#递归,将所有的分散的node连成一棵树
	for tensor_c in stem_output:
		dfs_search(tensor_c,stem,newPath,stem_tensor,end_tensor)
	stem_sort = sorted(stem,key = sort_key)
	end = stem_sort[-1][-1]
	
	node_dict={}
	#构建二叉树,利用node_dict [C张量:[A张量，B张量]] 来隐式的构建了二叉树
	for node in stem_sort:
		left_child = node[0]
		right_child = node[1]
		children = [left_child,right_child]
		node_dict[node[2]] = children
	#stems 中包含所有的stem，并且顺序也是对的[stem0,stem1,......]  stem0 = [[1,2,3],[3,4,5],[5,6,7]......]
	stems=[]

	main_stem = dfs_search_stem(node_dict,stems,end,indices,trunk_num)
	#print("main_stem is ",main_stem)
	# for node in main_stem:
	# 	stems.append(node)
	if len(main_stem) > 0:
		stems.append(main_stem)

	return stems

#挑出stems里长度超过3的stem，并返回下每一段的起始path序号和每段的长度
def Get_stem_info(stems,path):
	start_pos = []
	stem_length = []

	for stem in stems:
		if len(stem) > 3:
			stem_length.append(len(stem))
			pos = path.index( stem[0] )
			start_pos.append( pos )
	return start_pos,stem_length

def Check_stem_legal(stems,indices):
	for stem in stems:
		if len(stem) >= 3:
			for i in range(len(stem) - 1):

				if len(indices[stem[i][0]]) > 13 or len(indices[stem[i][1]]) < 13:
					print("dimenson error at : ",stem[i])
				if stem[i][2] != stem[i+1][1]:
					print("stem is not continous at : ",stem[i])


#动归挑出stems里长度超过3的stem，并返回下每一段的起始path序号和每段的长度
def pick_realStem(stems):
	start_pos=[]
	all_stem_length=[]
	stem_real=[]

	for i,stem in enumerate(stems):
		all_stem_length.append(len(stem))
		
		if len(stem) >= 3:
			stem_real.append(i)
		if i == 0:
			start_pos.append(0)
			continue	
		else:
			start_pos.append( start_pos[i-1] + all_stem_length[i-1] )
	
	stem_start_in_path = []
	stem_length = []

	for real_stem in stem_real:
		stem_start_in_path.append( start_pos[real_stem] )
		stem_length.append( all_stem_length[real_stem])
	
	return stem_start_in_path, stem_length

#该方法用于剪枝，使整个path只包含一个stem，弃用
def detect_stem_szq(newPath,indices,target_dim):
	stem = copy.deepcopy(newPath) 
	#原始stem上每一步收缩运算输出的tensor
	stem_output = []
	for step in newPath:
		if len(indices[step[0]]) < target_dim and len(indices[step[1]]) < target_dim:
			stem.remove(step)
	#若stem上的节点收缩生成的张量和它的队友的维度小于14，则需要连接操作,需递归的去找
	#先把所有的输出的张量小于14维的添加进stem_output中去,将所有的已经在stem上的tensor加入到stem_tensor中
	
 
	#bug here  这颗收缩树的终点不一定是stem[-1][-1]
	end_tensor = stem[-1][-1]

	stem_tensor = []
	for path in stem:
		stem_tensor.append(path[0])
		stem_tensor.append(path[1])
		if len(indices[path[2]]) < target_dim:
			stem_output.append(path[2])
	#stem_out用于存储容易出问题的张量的id
	#然后再遍历stem_output,查看与该tensor做收缩的张量是否也是小于14维的,若大于14维说明这个张量没问题，将其从stem_out中删除
	for tensor_a in stem_output:
		tensor_b = 0
		for step in newPath:
			#只检查前两个元素
			if tensor_a in step[:2]:
				tensor_a_index = step.index(tensor_a)
				if tensor_a_index == 0:
					tensor_b = step[1]
				else:
					tensor_b = step[0]
				#若与做a收缩的张量b大于等于14维，则将a从stem_out中移除
				if len(indices[tensor_b]) >= target_dim:
					stem_output.remove(tensor_a)

	#递归
	for tensor_c in stem_output:
		dfs_search(tensor_c,stem,newPath,stem_tensor,end_tensor)
	stem_sort = sorted(stem,key = sort_key)
	end = stem_sort[-1][-1]
	# #return stem_sort
	#这里是在去除branch
	for step in stem_sort:
		start_index = stem_sort.index(step)
		if step[-1] != end:
			while(stem_sort[start_index + 1][0] != step[-1] and stem_sort[start_index + 1][1] != step[-1]):
				del stem_sort[start_index + 1]
	return stem_sort


def sort_key(path):
	return path[-1]


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
	
	file.write(str(newPath))
	file.write('\n')
	file.write("original time cost is :"+str(original_time))
	file.write('\n')
	file.write("slice time cost is :"+str(slice_time))
	file.write('\n')
	file.write(str(get_contraction_from_tree(tree_s)))
	file.close()
	

	return input_list, slice_list, slice_tensor_id, indices_after_slice

def write2slicedtxt(mpath, eq_original, eq_sliced, slice_list, newPath, stem_start, stem_length, tree_s, info):
	filename = mpath+'/sliced.txt'
	slice_time = tree_s.contraction_cost()
	file = open(filename,'w')
	input_subscripts = info.input_subscripts
	
	
	input_list = input_subscripts.split(',')
	#导出未slice前的tensor的符号字符串
	for item in input_subscripts:
		file.write(item)
	file.write('\n')
	slice_tensor_id = []

    #找出被slice的tensor的id
	for slice_index in slice_list:
		for i,item in enumerate(eq_original):
			if slice_index in item:
				slice_tensor_id.append(i)
	slice_tensor_id = list(set(slice_tensor_id))
	slice_tensor_id.sort()
	
	#导出被slice的tensor的id
	
	file.write(str(slice_tensor_id))
	file.write('\n')
	file.write(str(slice_list))
	file.write('\n')

	for item in eq_sliced:
		file.write(item)
		file.write(',')
	file.write('\n')
 
	file.write(str(newPath))
	file.write('\n')

	file.write(str(stem_start))
	file.write('\n')

	file.write(str(stem_length))
	file.write('\n')


 
    
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
def write2array(mpath,tn):
	filename = mpath+'/array.txt'
	
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
	#开openleg
	rehs = circ.amplitude_rehearse(b=basis,optimize=opt)
	print(type(circ))
	tn = rehs['tn']
	print(rehs)
	info = rehs['info']
	write2array(mpath,tn)
	
	tree_s = opt.get_tree()
	trees_path = tree_s.get_path()
	slice_list = tree_s.sliced_inds
	
	#利用path建一棵没有slice的完整的收缩树
	tree_original = ctg.ContractionTree.from_path(inputs = tree_s.inputs,output=tree_s.output,size_dict=tree_s.size_dict,path=trees_path)

	
	input_subscripts = info.input_subscripts
	input_list = input_subscripts.split(',')
	eq_after_slice=[]
	for string in input_list:
		for char in slice_list:
			if char in string:
				string = string.replace(char,'')
		eq_after_slice.append(string)
	
	lhs_sliced = eq_after_slice
	newPath = path_str_to_int(trees_path)
	indices = T.get_newPath_eq_index(lhs_sliced, newPath)
	original_path = exchange_A_B_szq(newPath,indices)
	szq_stems=find_stems(original_path,indices,13) # LDM can only store 13 dimension sub-tensor
	stems = detect_stems_szq_v2(original_path, indices, 13, 0)
	stem_start,stem_length = pick_realStem(stems)
	Stem=[]
	for stem in stems:
		for contract in stem:
			Stem.append(contract)

	szqPath,before_len = get_newpath_szq(original_path,Stem)
	for i in range(len(stem_start)):
		stem_start[i] = stem_start[i]+before_len
	
	#实验一下官方提供的两种官方提供的计算时间复杂度的方法
	#_,_,_,lhs_sliced = write2slicetxt_trans(mpath, tree_s, info, tree_original.contraction_cost(),trees_path)
	write2slicedtxt(mpath, input_list, eq_after_slice, slice_list, szqPath, stem_start, stem_length, tree_s, info)
	
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
	
	
