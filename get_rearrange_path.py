import os
import sys
import math
import copy
from  trans import Trans as T
from trans import Test  

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
	

if __name__=='__main__':

	if len(sys.argv) != 2:
		print('Usage: get_rearrange_path.py filename')
		exit(-1)

	#output = 'ǒıøâĥQPßÜbawvfeEDAzjiÂÁMLIHnmħĝrqÎÍāİįĬÚÙÒÑÖÕÊÉÆÅYXUT'
	#ix_out = set()
	#for i in output:
	#	ix_out.add(i)

	test = Test()

	f=open(str(sys.argv[1]))
	datas = f.readlines()
	idx = datas[3]
	idx = idx.strip()
	
	print("over")
	eq = idx
	idx_bk = idx
	idx = idx.split(',')

	index = datas[1]
	index = index.strip()
	index = index.replace('(','')
	index = index.replace(')','')
	index = index.replace(' ','')
	index = index.replace("'",'')
	sl = ''.join(index.split(','))
	ix = set()
	for i in sl:
		ix.add(i)

	f.close()

	print("index is:",index)

	#print(eq)
	#for i in ix_out:
	#	eq = eq.replace(i,'')

	lhs = eq
	lhs = lhs.split(',')
	change_index = set()
	for i in ix:
		for j in range(len(lhs)):
			if (lhs[j].find(i) >= 0):
				change_index.add(j)
	#print(sorted(change_index))
	#print(ix)

	eq_sliced = eq
	

	lhs_sliced = eq_sliced.split(',')
	path = get_last_line(str(sys.argv[1]))

	print("path is:",path)
	
	#原始的三元组path
	newPath = T.trans_path_index_str(path)
	print("original path is:",newPath)
	print("original path length is ",len(newPath))
	indices = T.get_newPath_eq_index(lhs_sliced, newPath)

	#利用szq_exchange交换A、B张量位置后的path
	original_path = exchange_A_B_szq(newPath,indices)

	szq_stems=find_stems(original_path,indices,13)
	for stem in szq_stems:
		print("each stem :",stem)
	
	print("indices")
	#print(len(indices[107]),",",len(indices[547]))
	#寻找path中的所有stems 
	trunk_num = 0
	stems = detect_stems_szq_v2(original_path, indices, 13, trunk_num)
	for stem in stems:
		print(stem)
		test.test_stem(stem)
	
	Check_stem_legal(stems,indices)
	test.test_new_path(original_path, True)
	#print(indices)
	#for index in indices:
	#	print(indices.index(index), len(index))
	#tail tensor original index
	#print("stems is:",stems)
	stem_start,stem_length = pick_realStem(stems)
	
	
	print("new path is")
	Stem=[]
	for stem in stems:
		for contract in stem:
			Stem.append(contract)
	#利用get_newpath_szq 将path重排，将stem都放到合适的位置
	szqPath,before_len = get_newpath_szq(original_path,Stem)
	print("Stem length is : ",len(Stem))
	print("szq path length is ",len(szqPath))
	print("szq path is:",szqPath)
    
	#计算绝对位置
	for i in range(len(stem_start)):
		stem_start[i] = stem_start[i]+before_len
	print("stem_length is:",stem_length)
	print("stem_start is:",stem_start)
	
	stem_start,stem_length = Get_stem_info(stems,szqPath)
	print("stem_length is:",stem_length)
	print("stem_start is:",stem_start)
	print("trunk_num is: ",trunk_num)
	rearrange_path, st = T.path_rearrange(original_path, indices, Stem, [])
	
	start_pos = rearrange_path.index(st)
	
	T.exchange_A_and_B(rearrange_path, indices, start_pos)
	#print(st, start_pos)
	#rearrange path
	
	
	
	newPath = rearrange_path
	#szq  这里又rearrange了一下
	T.path_index_rearrange(newPath, indices)
	
	# print(newPath)
	test.test_new_path(newPath, True)
	path_int = T.retrans_path_index(newPath)
	
	#print("new path is:",newPath)
	#print(indices)
	#for index in indices:
	#	print(indices.index(index), len(index))
	#print(path_int)
	for i in range(len(newPath)+1,2*len(newPath)+1):
		lhs_sliced.append('')
	for i in range(len(newPath)):
		ct1 = newPath[i][0]
		ct2 = newPath[i][1]
		common_index = [x for x in lhs_sliced[ct1] if x in lhs_sliced[ct2]]
		res_index = ''.join(y for y in (lhs_sliced[ct1]+lhs_sliced[ct2]) if y not in common_index)
		res_index = res_index.replace("'",'')
		lhs_sliced[newPath[i][2]] = res_index
	#get trans map
#	print(newPath)
	#print(lhs_sliced)
	trans_map = T.gen_trans_map(newPath, lhs_sliced)
	# for i in range(len(newPath)):
	# 	print(i, trans_map[i])
	end_pos = len(newPath)
	pos, comms, comm_len, st_pos, ed_pos = T.detect_fused_pos(newPath, indices, end_pos, start_pos, 0, False)	
	print("fused pos is",pos)
	tensor_dim = []
	for i in range(st_pos, ed_pos):
		tensor_dim.append(len(indices[newPath[i][0]]))
		tensor_dim.append(len(indices[newPath[i][1]]))
		tensor_dim.append(len(indices[newPath[i][0]] & indices[newPath[i][1]])) 
	print(comm_len)
	fused_map, step_len = T.gen_whole_fused_map(newPath, lhs_sliced, comms, pos, False)
	#step length:融合段的数量，和fused map长度是对的上的
	print("step length is :" ,(step_len))
	print("tensor dim is :",len(tensor_dim))
	
	
