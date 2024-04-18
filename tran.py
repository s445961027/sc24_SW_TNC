#!/usr/bin/python

import numpy as np
import copy
import sys

class Trans:
	def __init__(self):
		pass

	def list_to_tuple(self, path):
		
		for i in range(len(path)):
			path[i] = tuple(path[i])
	
		return tuple(path)

	def get_newPath_index(self, info, newPath):
	
	    index = list()
	    lhs, rhs = info.eq.replace(' ', '').split('->')
	    lhs = lhs.split(',')
	    for i in range(len(lhs)):
	        index.append(set(lhs[i]))
	
	    for i in range(len(newPath)):
	        index.append((index[newPath[i][0]] | index[newPath[i][1]]) - (index[newPath[i][0]] & index[newPath[i][1]]))
	
	    return index

	def trans_path_tuple_index(self, path_info):
	
	    path_str=str(path_info)
	    path = path_str.split("), (")
	    path[0] = path[0].strip(" ((")
	    path[len(path) - 1] = path[len(path) - 1].strip("))\n")
	    steps = len(path)
	    k = 1
	    path_int = list()
	    for step in path:
	        step = step.split(", ")
	        step_new = [int(step[0]), int(step[1]), steps - k]
	        path_int.append(step_new)
	        k += 1
	    path = path_int
	
	    align = [0]*steps
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

	def trans_path_index(self, path):
		steps = len(path)
		align = [0]*steps
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
	def retrans_path_index_sxm(self, newPath):
		steps = len(newPath)
		align = [0]*(steps*2)
		path = list()
		k = 1 
		for step in newPath:
			step_new = copy.deepcopy(step)
			step_new[0] = step[0] - align[step[0]]
			step_new[1] = step[1] - align[step[1]]
			step_new[2] = steps - k 
			path.append([min(step_new[0],step_new[1]),max(step_new[0],step_new[1])])
			for i in range(step[0], steps*2):
				align[i] += 1
			for i in range(step[1], steps*2):						
				align[i] += 1
			k += 1
		return path
	def retrans_path_index(self, newPath):
		steps = len(newPath)
		align = [0]*(steps*2)
		path = list()
		k = 1
		for step in newPath:
			step_new = copy.deepcopy(step)
			step_new[0] = step[0] - align[step[0]]
			step_new[1] = step[1] - align[step[1]]
			step_new[2] = steps - k
			#path.append(step_new)
			path.append([step_new[0],step_new[1]])
			for i in range(step[0], steps*2):
				align[i] += 1
			for i in range(step[1], steps*2):
				align[i] += 1
			k += 1
		return path
	
	def path_index_rearrange(self, newPath, indices):
		steps = len(newPath)
		for i in range(steps):
			temp = newPath[i][2]
			rep = steps + i + 1
			if temp == rep:
				continue
			newPath[i][2] = -1
			index = [0, 0]
			for j in range(steps):
				if newPath[j][0] == temp:
					newPath[j][0] = -1
					index = [j, 0]
					break
				elif newPath[j][1] == temp:
					newPath[j][1] = -1
					index = [j, 1]
					break
			for step in newPath:
				if step[0] == rep:
					step[0] = temp
					break
				elif step[1] == rep:
					step[1] = temp
					break
				if step[2] == rep:
					step[2] = temp
			newPath[i][2] = rep
			newPath[index[0]][index[1]] = rep
			indices[temp], indices[rep] = indices[rep], indices[temp]
	
		for i in range(steps):
			if newPath[i][0] > newPath[i][1]:
				step[0], step[1] = step[1], step[0]
	'''
		for i in range(steps):
			if newPath[i][0] > newPath[i][1]:
				step = newPath[i]
				ind_1 = -1
				ind_2 = -1
				for j in range(i):
					if newPath[j][2] == step[0]:
						ind_1 = j
					if newPath[j][2] == step[1]:
						ind_2 = j
				if ind_1 != -1:
					newPath[ind_1][2] = step[1]
				if ind_2 != -1:
					newPath[ind_2][2] = step[0]
				step[0], step[1] = step[1], step[0]
				indices[step[0]], indices[step[1]] = indices[step[1]], indices[step[0]]
	'''
	def trans_path_to_tree(self, newPath):
#newPath is a path with none node moved after contraction
		steps = len(newPath)
		nodeset = [0]*(2*steps + 1)
		children = dict()
		for step in newPath:
			if step[0] <= steps:
				nodeset[step[0]] = frozenset([step[0]])
			if step[1] <= steps:
				nodeset[step[1]] = frozenset([step[1]])
			nodeset[step[2]] = nodeset[step[0]] | nodeset[step[1]]
			children.setdefault(nodeset[step[2]], (nodeset[step[0]], nodeset[step[1]]))
		return children

	def detect_stem(self, newPath, indices, target_dim):
		stem = copy.deepcopy(newPath)
		for step in newPath:
			if len(indices[step[0]]) < target_dim and len(indices[step[1]]) < target_dim:
			#if (((len(indices[step[0]]) > target_dim and len(indices[step[1]]) < target_dim) or (len(indices[step[0]]) > target_dim and len(indices[step[1]])<target_dim ))!=1):
				stem.remove(step)
		return stem

	def reverse_stem(self, stem, indices, newPath):
		for i in range(len(stem)-1):
			if stem[i][2] != stem[i+1][0] and stem[i][2] != stem[i+1][1]:
				for j in range(i+1, len(stem)):
					if (stem[j][0] == stem[i][2] or stem[j][1] == stem[i][2]):
						stem[i+1:] = list(reversed(stem[i+1:]))
						index_1 = newPath.index(stem[i+1])
						gen = stem[i+1][0]
						if stem[i+1][0] in stem[i]:
							stem[i+1][1], stem[i+1][2] = stem[i+1][2], stem[i+1][1]
							gen = stem[i+1][1]
						elif stem[i+1][1] in stem[i]:
							stem[i+1][0], stem[i+1][2] = stem[i+1][0], stem[i+1][1]
						newPath[index_1] = copy.deepcopy(stem[i+1])
						for k in range(i+2, len(stem)-1):
							index = newPath.index(stem[k])
							if stem[k][0] == stem[k+1][2]:
								stem[k][0], stem[k][2] = stem[k][2], stem[k][0]
							else:
								stem[k][1], stem[k][2] = stem[k][2], stem[k][1]
							newPath[index] = copy.deepcopy(stem[k])
						index_2 = newPath.index(stem[len(stem)-1])
						if stem[len(stem)-1][0] < len(newPath) + 1:
							stem[len(stem)-1][1], stem[len(stem)-1][2] = stem[len(stem)-1][2], stem[len(stem)-1][1]	
						elif stem[len(stem)-1][1] < len(newPath) + 1:
							stem[len(stem)-1][0], stem[len(stem)-1][2] = stem[len(stem)-1][2], stem[len(stem)-1][0]	
						elif len(indices[stem[len(stem)-1][0]]) < len(indices[stem[len(stem)-1][1]]):
							stem[len(stem)-1][0], stem[len(stem)-1][2] = stem[len(stem)-1][2], stem[len(stem)-1][0]
						else:
							stem[len(stem)-1][1], stem[len(stem)-1][2] = stem[len(stem)-1][2], stem[len(stem)-1][1]	
						invo = stem[len(stem)-1][2]
						newPath[index_2] = copy.deepcopy(stem[len(stem)-1])
						k_1 = index_1
						for k in range(index_1+1, len(newPath)):
							if 2*len(newPath) == newPath[k][2]:
								ind_1 = newPath[k_1].index(gen)
								if gen == newPath[k][0]:
									newPath[k_1][ind_1] = newPath[k][1]
								if gen == newPath[k][1]:
									newPath[k_1][ind_1] = newPath[k][0]
								del newPath[k]
								break
							if gen == newPath[k][0]:
								newPath[k][0], newPath[k][2] = newPath[k][2], newPath[k][0]
								gen = newPath[k][0]
								k_1 = k
							elif gen == newPath[k][1]:
								newPath[k][1], newPath[k][2] = newPath[k][2], newPath[k][1]
								gen = newPath[k][1]
								k_1 = k

						for k in range(index_2-1, 0, -1):
							if invo == newPath[k][2]:
								if newPath[k][0] < len(newPath) + 2 and newPath[k][1] < len(newPath) + 2:
									new_step = [newPath[k][0], gen, 2*(len(newPath)+1)]
									newPath.append(new_step)
									newPath[k] = [newPath[k][1], newPath[k][2], gen]
									indices[gen] = indices[newPath[k][0]] ^ indices[newPath[k][1]]
									break
								if newPath[k][1] < len(newPath) + 2:
									newPath[k][0], newPath[k][2] = newPath[k][2], newPath[k][0]
									invo = newPath[k][2]
								elif newPath[k][0] < len(newPath) + 2:
									newPath[k][1], newPath[k][2] = newPath[k][2], newPath[k][1]
									invo = newPath[k][2]
								elif len(indices[newPath[k][0]]) < len(indices[newPath[k][1]]):
									newPath[k][0], newPath[k][2] = newPath[k][2], newPath[k][0]
									invo = newPath[k][2]
								else:	
									newPath[k][1], newPath[k][2] = newPath[k][2], newPath[k][1]
									invo = newPath[k][2]
						break
				break
		return stem, newPath

	def path_contraction_cost(newPath, indices):
		cost = 0
		for step in newPath:
			dA = len(indices[step[0]])
			dB = len(indices[step[1]])
			dC = len(indices[step[2]])
			cost += 2**(int((dA + dB + dC)/2))
		return cost

	def stem_to_path(self, stem, newPath, indices):
		finalPath = list()
		steps = len(newPath)
		stem_len = len(stem)
		formed = [1]*(steps+1) + [0]*steps
		used = [0]*(2*steps+1)
		for step in stem:
			used[step[0]] = 1
			used[step[1]] = 1
			if formed[step[0]] == 0:
				self.former(newPath, step, finalPath, 0, formed, used)
			if formed[step[1]] == 0:
				self.former(newPath, step, finalPath, 1, formed, used)
			formed[step[2]] = 1
			finalPath.append(copy.deepcopy(step))
		for i in range(2*steps):
			if used[i] == 0:
				self.user(newPath, i, finalPath, formed, used)
		if len(finalPath) != len(newPath):
			print("wrong path with length " + str(len(finalPath)))
			
		return finalPath

	def former(self, newPath, step, finalPath, tag, formed, used):
		for contr in newPath:
			if contr[2] == step[tag]:
				if formed[contr[0]] == 0:
					self.former(newPath, contr, finalPath, 0, formed, used)
				if formed[contr[1]] == 0:
					self.former(newPath, contr, finalPath, 1, formed, used)
				formed[step[tag]] = 1
				finalPath.append(copy.deepcopy(contr))
				used[contr[0]] = 1
				used[contr[1]] = 1
				break

	def user(self, newPath, index, finalPath, formed, used):
		for contr in newPath:
			if contr[0] == index:
				if formed[contr[1]] == 0:
					self.former(newPath, contr, finalPath, 1, formed, used)
				finalPath.append(copy.deepcopy(contr))
				used[contr[0]] = 1
				used[contr[1]] = 1
				formed[contr[2]] = 1
				break
			if contr[1] == index:
				if formed[contr[0]] == 0:
					self.former(newPath, contr, finalPath, 0, formed, used)
				finalPath.append(copy.deepcopy(contr))
				used[contr[0]] = 1
				used[contr[1]] = 1
				formed[contr[2]] = 1
				break
					
	def find_critical_tensors(self, lifetime, target):
		critical = []
		for node in lifetime:
			if len(node[1]) == target:
				critical.append(copy.deepcopy(node))
		return critical

	def replace_candidate_from_criticals(self, critical):
		if len(critical) > 0:
			candidate = set() | critical[0][1]
			for i in range(1, len(critical)):
				temp = candidate & critical[i][1]
				candidate = temp
			return candidate
		return []

	def branch_merge(self, stem, indices, newPath, pos):
		if stem[pos][2] != stem[pos+1][0] and stem[pos][2] != stem[pos+1][1]:
			print("stem wrong!")

		i1 = newPath.index(stem[pos])
		i2 = newPath.index(stem[pos+1])
		diff = stem[pos+1][2] - stem[pos][2]
		lower = stem[pos][2]
		upper = stem[pos+1][2]
		stem_pos = stem[pos+1].index(stem[pos][2])
		bran_pos = 1 - stem_pos
		if diff != i2 - i1:
			print("path wrong")
		for i in range(i1, i2-1):
			newPath[i][2] += 1
			newPath[i+1][2] -= 1
			if newPath[i+1][0] > lower:
				newPath[i+1][0] -= 1
			if newPath[i+1][1] > lower:
				newPath[i+1][1] -= 1
			newPath[i], newPath[i+1] = newPath[i+1], newPath[i]
			indices[newPath[i][2]] = indices[newPath[i][0]] ^ indices[newPath[i][1]]
		newPath[i2][stem_pos] = newPath[i2-1][2]
		if newPath[i2][bran_pos] > len(newPath):
			newPath[i2][bran_pos] -= 1
		stem_p = int(len(indices[stem[pos][0]]) < len(indices[stem[pos][1]]))
		newPath[i2-1][stem_p], newPath[i2][bran_pos] = newPath[i2][bran_pos], newPath[i2-1][stem_p]
		if newPath[i2-1][0] > newPath[i2-1][1]:
			newPath[i2-1][0], newPath[i2-1][1] = newPath[i2-1][1], newPath[i2-1][0]
		if newPath[i2][0] > newPath[i2][1]:
			newPath[i2][0], newPath[i2][1] = newPath[i2][1], newPath[i2][0]
		indices[newPath[i2-1][2]] = indices[newPath[i2-1][0]] ^ indices[newPath[i2-1][1]]
		indices[newPath[i2][2]] = indices[newPath[i2][0]] ^ indices[newPath[i2][1]]
		stem[pos] = newPath[i2-1]
		stem[pos+1] = newPath[i2]

	def branch_exchange(self, stem, indices, newPath, pos):
		if stem[pos][2] != stem[pos+1][0] and stem[pos][2] != stem[pos+1][1]:
			print("stem wrong!")
		
		i1 = newPath.index(stem[pos])
		i2 = newPath.index(stem[pos+1])
		diff = stem[pos+1][2] - stem[pos][2]
		lower = stem[pos][2]
		upper = stem[pos+1][2]
		stem_pos = stem[pos+1].index(stem[pos][2])
		bran_pos = 1 - stem_pos
		if diff != i2 - i1:
			print("path wrong")
		for i in range(i1, i2-1):
			newPath[i][2] += 1
			newPath[i+1][2] -= 1
			if newPath[i+1][0] > lower:
				newPath[i+1][0] -= 1
			if newPath[i+1][1] > lower:
				newPath[i+1][1] -= 1
			newPath[i], newPath[i+1] = newPath[i+1], newPath[i]
			indices[newPath[i][2]] = indices[newPath[i][0]] ^ indices[newPath[i][1]]
		newPath[i2][stem_pos] = newPath[i2-1][2]
		if newPath[i2][bran_pos] > len(newPath):
			newPath[i2][bran_pos] -= 1
		stem_p = int(len(indices[stem[pos][0]]) < len(indices[stem[pos][1]]))
		bran_p = 1 - stem_p
		newPath[i2-1][bran_p], newPath[i2][bran_pos] = newPath[i2][bran_pos], newPath[i2-1][bran_p]
		if newPath[i2-1][0] > newPath[i2-1][1]:
			newPath[i2-1][0], newPath[i2-1][1] = newPath[i2-1][1], newPath[i2-1][0]
		if newPath[i2][0] > newPath[i2][1]:
			newPath[i2][0], newPath[i2][1] = newPath[i2][1], newPath[i2][0]
		indices[newPath[i2-1][2]] = indices[newPath[i2-1][0]] ^ indices[newPath[i2-1][1]]
		indices[newPath[i2][2]] = indices[newPath[i2][0]] ^ indices[newPath[i2][1]]
		stem[pos] = newPath[i2-1]
		stem[pos+1] = newPath[i2]

class Test:
	def __init__(self):
		pass

	def test_stem(self, stem):
		for i in range(len(stem) - 1):
			if stem[i][2] != stem[i+1][0] and stem[i][2] != stem[i+1][1]:
				print("stem is not linear! at "+ str(i))

	def test_new_path(self, newPath, deep_check):
		used = [0]*(2*len(newPath))
		formed = [1]*(len(newPath)+1) + [0]*len(newPath)
		for step in newPath:
			if deep_check:
				if formed[step[2]] >= 1:
					print("redundant gen at ", step, step[2], formed[step[2]])
				if used[step[0]] >= 1:
					print("redundant use at ", step, step[0], used[step[0]])
				if used[step[1]] >= 1:
					print("redundant use at ", step, step[1], used[step[1]])
				if formed[step[0]] == 0:
					print("haven't formed at ", step, step[0])
				if formed[step[1]] == 0:
					print("haven't formed at ", step, step[1])
			else:
				if formed[step[2]] >= 1:
					print("redundant gen at ", step, step[2], formed[step[2]])
				if used[step[0]] >= 1:
					print("redundant use at ", step, step[0], used[step[0]])
				if used[step[1]] >= 1:
					print("redundant use at ", step, step[1], used[step[1]])
			used[step[0]] += 1
			used[step[1]] += 1
			formed[step[2]] += 1
	
	def test_indices(self, indices, newPath):
		for i in range(len(indices)):
			if len(indices[i]) > 5 and i < len(indices) / 2:
				print("warning! "+ str(i) +" has " + str(len(indices[i])) + " indices")
		
		for step in newPath:
			expected = (indices[step[0]] | indices[step[1]]) - (indices[step[0]] & indices[step[1]])
			if expected != indices[step[2]]:
				print("indices wrong at step: ", step)
				if newPath.index(step) < 10:
					print(indices[step[0]], indices[step[1]], expected, indices[step[2]])


def load_data(filename):
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

def create_indices_set(newPath, data_tensor):
	steps = len(newPath)
	indices = [0]*(2*steps+1)
	indices[2*steps] = frozenset()
	for i in range(steps):
		if i != int(data_tensor[i][0]):
			print("wrong!")
		if indices[newPath[i][0]] == 0:
			indices[newPath[i][0]] = frozenset(data_tensor[i][3])
		if indices[newPath[i][1]] == 0:
			indices[newPath[i][1]] = frozenset(data_tensor[i][2])
		indices[newPath[i][2]] = frozenset(data_tensor[i][1])
	return indices

if __name__ == "__main__":
#print sys.getdefaultencoding()
#reload(sys)
#sys.setdefaultencoding('UTF-8')
	trans = Trans()
	test = Test()
	
	path_int, data_tensor = load_data("path.txt")
	print(type(path_int))
#print(data_tensor)
	newPath = trans.trans_path_index(path_int)
	
	print(newPath)
	test.test_new_path(newPath, True)
#print(newPath)
	indices = create_indices_set(newPath, data_tensor)
	test.test_indices(indices, newPath)
#print(indices)
	children = trans.trans_path_to_tree(newPath)
#print(children)

#print(newPath)
	stem = trans.detect_stem(newPath, indices, 20)
#	print(stem)
#	print(newPath)
	stem, newPath = trans.reverse_stem(stem, indices, newPath)
#	print(newPath)
#	print(stem)
#	trans.branch_exchange(stem, indices, newPath, 2)
#	test.test_indices(indices, newPath)
#print(stem)
#for i in range(5):
	trans.branch_merge(stem, indices, newPath, 0)
	test.test_indices(indices, newPath)
#print(stem)
	finalPath = trans.stem_to_path(stem, newPath, indices)
	test.test_new_path(finalPath, True)
#print(finalPath)	
	path = trans.retrans_path_index(newPath)
#print(path)
