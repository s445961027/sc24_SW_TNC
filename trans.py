#!/usr/bin/python

import numpy as np
import copy
import sys

class Trans:
	def __init__(self):
		pass

	@staticmethod
	def trans_path_index_str(path_info):
	
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

	@staticmethod
	def trans_path_index(path):
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

	@staticmethod
	def list_to_tuple(path):
	
		for i in range(len(path)):
			path[i] = tuple(path[i])

		return tuple(path)

	@staticmethod
	def get_newPath_index(info, newPath):
	
	    index = list()
	    lhs, rhs = info.eq.replace(' ', '').split('->')
	    lhs = lhs.split(',')
	    for i in range(len(lhs)):
	        index.append(set(lhs[i]))
	
	    for i in range(len(newPath)):
	        index.append((index[newPath[i][0]] | index[newPath[i][1]]) - (index[newPath[i][0]] & index[newPath[i][1]]))
	
	    return index
	
	@staticmethod
	def get_newPath_eq_index(inputs, newPath):
		index = list()
		print("inputs len is",len(inputs))
		for i in range(len(inputs)):
			index.append(set(inputs[i]))
		print("orign len is",len(index))
		for i in range(len(newPath)):
			#print(newPath[i][0]," set is:",len(index[newPath[i][0]]),"  ",newPath[i][1],"set is:",len(index[newPath[i][1]]),"  ",newPath[i][2],"set is:",len(diff_set))
			index.append((index[newPath[i][0]] | index[newPath[i][1]]) - (index[newPath[i][0]] & index[newPath[i][1]]))
		return index
	
	@staticmethod
	def get_newPath_eq_index_list(inputs, newPath):
		index = [0]*(2*len(newPath)+1)
		for i in range(len(inputs)):
			index[i] = list(inputs[i])
		for i in range(len(newPath)):
			index1 = copy.deepcopy(index[newPath[i][0]])
			index2 = copy.deepcopy(index[newPath[i][1]])
			index[newPath[i][2]] = list(set(index1) ^ set(index2))
		return index
					
	@staticmethod
	def exchange_A_and_B(newPath, indices, start_pos):
		if len(indices[newPath[start_pos][0]]) > len(indices[newPath[start_pos][1]]):
			newPath[start_pos][0], newPath[start_pos][1] = newPath[start_pos][1], newPath[start_pos][0]

		for i in range(start_pos+1, len(newPath)):
			#if len(indices[step[0]]) >= len(indices[step[1]]) and (len(indices[step[0]]) > 13 or len(indices[step[1]]) > 13):
			if newPath[i][0] == newPath[i-1][2]:
				newPath[i][0], newPath[i][1] = newPath[i][1], newPath[i][0]
#			if len(indices[step[0]]) <= 18 and len(indices[step[1]]) <= 18 and len(indices[step[0]]) >= len(indices[step[1]]):
#				if len(indices[step[0]]) >= 13 or len(indices[step[1]]) >= 13:
#					step[0], step[1] = step[1], step[0]

	@staticmethod
	def retrans_path_index_ordered(newPath):
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

	@staticmethod
	def retrans_path_index(newPath):
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
			path.append((step_new[0],step_new[1]))
			for i in range(step[0], steps*2):
				align[i] += 1
			for i in range(step[1], steps*2):
				align[i] += 1
			k += 1
		return tuple(path)

	@staticmethod
	def path_index_rearrange(newPath, indices):
		keys = [i for i in range(2*len(newPath)+1)]
		for i in range(len(newPath)):
			keys[newPath[i][2]] = i + len(newPath)+1
			#keys[i+len(newPath)+1] = newPath[i][2]
		#print(keys)
		for i in range(len(newPath)):
			newPath[i][0] = keys[newPath[i][0]]
			newPath[i][1] = keys[newPath[i][1]]
			newPath[i][2] = keys[newPath[i][2]]
			indices[newPath[i][2]] = indices[newPath[i][0]]^indices[newPath[i][1]]
		
		'''
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

	@staticmethod
	def trans_path_to_tree(newPath):
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

	@staticmethod
	def detect_stem(newPath, indices, target_dim):
		stem = copy.deepcopy(newPath)
		for step in newPath:
			if len(indices[step[0]]) < target_dim and len(indices[step[1]]) < target_dim:
			#if (((len(indices[step[0]]) > target_dim and len(indices[step[1]]) < target_dim) or (len(indices[step[0]]) > target_dim and len(indices[step[1]])<target_dim ))!=1):
				stem.remove(step)
		return stem
	

	

	

	@staticmethod
	def reverse_stem(stem, indices, newPath):
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

#	print(stem)

	@staticmethod
	def stem_to_path(stem, newPath, indices):
		finalPath = list()
		steps = len(newPath)
		stem_len = len(stem)
		formed = [1]*(steps+1) + [0]*steps
		used = [0]*(2*steps+1)
		for step in stem:
			used[step[0]] = 1
			used[step[1]] = 1
			if formed[step[0]] == 0:
				Trans.former(newPath, step, finalPath, 0, formed, used)
			if formed[step[1]] == 0:
				Trans.former(newPath, step, finalPath, 1, formed, used)
			formed[step[2]] = 1
			finalPath.append(copy.deepcopy(step))
		for i in range(2*steps):
			if used[i] == 0:
				Trans.user(newPath, i, finalPath, formed, used)
		if len(finalPath) != len(newPath):
			print("wrong path with length " + str(len(finalPath)))
			
		return finalPath

	@staticmethod
	def former(newPath, step, finalPath, tag, formed, used):
		for contr in newPath:
			if contr[2] == step[tag]:
				if formed[contr[0]] == 0:
					Trans.former(newPath, contr, finalPath, 0, formed, used)
				if formed[contr[1]] == 0:
					Trans.former(newPath, contr, finalPath, 1, formed, used)
				formed[step[tag]] = 1
				finalPath.append(copy.deepcopy(contr))
				used[contr[0]] = 1
				used[contr[1]] = 1
				break

	@staticmethod
	def user(newPath, index, finalPath, formed, used):
		for contr in newPath:
			if contr[0] == index:
				if formed[contr[1]] == 0:
					Trans.former(newPath, contr, finalPath, 1, formed, used)
				finalPath.append(copy.deepcopy(contr))
				used[contr[0]] = 1
				used[contr[1]] = 1
				formed[contr[2]] = 1
				break
			if contr[1] == index:
				if formed[contr[0]] == 0:
					Trans.former(newPath, contr, finalPath, 0, formed, used)
				finalPath.append(copy.deepcopy(contr))
				used[contr[0]] = 1
				used[contr[1]] = 1
				formed[contr[2]] = 1
				break
					

	@staticmethod
	def branch_merge(stem, indices, newPath, pos):
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
		stem[pos+1] = copy.deepcopy(newPath[i2])
		del stem[pos]

	@staticmethod
	def branch_exchange(stem, indices, newPath, pos):
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

	@staticmethod
	def path_rearrange(newPath, indices, stem, open_tensor):
		start_point = stem[-1]
		stemex = []
		index = newPath.index(start_point)
		stem_tensor = start_point[2]
		path = []
		open_pos = []
		#这段for循环无用
		for i in range(index+1):
			if newPath[i][0] in open_tensor or newPath[i][1] in open_tensor:
				open_pos.append(i)
				stemex.append(newPath[i])

		for i in range(index+1, len(newPath)):
			if newPath[i][0] == stem_tensor or newPath[i][1] == stem_tensor:
				stemex.append(newPath[i])
				stem_tensor = newPath[i][2]
			elif newPath[i][0] in open_tensor or newPath[i][1] in open_tensor:
				open_pos.append(i)
				stemex.append(newPath[i])
		for index in open_pos:
			open_ten = newPath[index][2]
			for i in range(index + 1, len(newPath)):
				if newPath[i][0] == open_ten or newPath[i][1] == open_ten:
					if newPath[i] not in stem and newPath[i] not in stemex:
						stemex.append(newPath[i])
						open_ten = newPath[i][2]
		#这里的问题
		for step in newPath:
			if step not in stem and step not in stemex:
				path.append(copy.deepcopy(step))
		#print(path)
		for step in newPath:
			if step not in path:
				path.append(copy.deepcopy(step))
		return path, stem[0]

	@staticmethod
	def gen_trans_map(newPath, indice_list):
		trans_map = []
		for step in newPath:
			trans_map_step = [list(range(len(indice_list[step[0]]))), list(range(len(indice_list[step[1]])))]
			comm = []
			#print(newPath.index(step), ": ", trans_map_step)
			for i in range(len(indice_list[step[0]])):
				if indice_list[step[0]][i] in indice_list[step[1]]:
					comm.append([i, indice_list[step[1]].index(indice_list[step[0]][i])])
			trans_map_step_temp = []
			for elem in comm:
				trans_map_step[0].remove(elem[0])
				trans_map_step[0].append(elem[0])
				trans_map_step[1].remove(elem[1])
				trans_map_step_temp.append(elem[1])
			trans_map_step[1] = trans_map_step_temp + trans_map_step[1]
			trans_map.append(trans_map_step)
		return trans_map
	
	@staticmethod
	def ldm_slice(newPath, indices, ed_pos=0, st_pos=0, axis=0, rma=False, AT=False):
		comm = []
		length = []
		pos = 0
		dimension = 13
		if rma:
			dimension = 19
		if axis == 0:
			st_indices = indices[newPath[st_pos][1]]
			length.append(len(st_indices))
			for i in range(st_pos+1, ed_pos):
				tar = i + 1
				while len(indices[newPath[tar][1]]) < 7 and tar < ed_pos:
					tar += 1
				comm = st_indices & indices[newPath[tar][1]]
				#print(newPath[i], len(indices[newPath[i][0]]), len(indices[newPath[i][1]]), len(indices[newPath[i][2]]))
				#print("step: ", i, "tar: ", tar, " , comm: ", comm, " , len: ", len(comm))
				#print(len(st_indices & indices[newPath[i][1]]))
				#print(len(st_indices), len(indices[newPath[i][1]]))
				length.append(len(indices[newPath[i][1]]))
				if len(comm) + dimension < max(length) or len(comm) + dimension < len(indices[newPath[tar][1]]):
				#if len(comm) + dimension < max(length):
					comm = st_indices & indices[newPath[i][1]]
					#print("fused end at ", i, " , slice: ", len(comm))
					pos = i
					break
			if pos == 0:
				pos = ed_pos
		#bug 待修，map中comm和branch会相交，奇怪
		elif axis == 1:
			ed_indices = indices[newPath[ed_pos-1][1]]
			length.append(len(ed_indices))
			for i in range(ed_pos-2, st_pos-1, -1):
				#tar = max(i - 1, st_pos)
				tar = i - 1
				while len(indices[newPath[tar][1]]) < 7 and tar > st_pos:
					tar -= 1
				comm = ed_indices & indices[newPath[tar][1]]
				print(newPath[i], len(indices[newPath[i][0]]), len(indices[newPath[i][1]]))
				print("step: ", i, "tar: ", tar, " , comm: ", comm, " , len: ", len(comm))
				print(len(ed_indices & indices[newPath[i][1]]))
				print(len(ed_indices), len(indices[newPath[i][1]]))
				length.append(len(indices[newPath[i][1]]))
				if len(comm) + dimension < max(length):
					pos = i
					comm = ed_indices & indices[newPath[i][1]]
					break		
			if pos == 0:
				pos = st_pos
		return pos, comm

	@staticmethod
	def detect_fused_pos(newPath, indices, ed_pos, st_pos=0, axis=0, rma=False):
		start_pos = st_pos
		end_pos = ed_pos
		comm = []
		comm_len = []
		for i in range(st_pos, len(newPath)):
			if len(indices[newPath[i][0]]) > 13 or len(indices[newPath[i][1]]) > 13:
				start_pos = i
				break
		for i in range(ed_pos-1, -1, -1):
			if len(indices[newPath[i][0]]) >= 13 or len(indices[newPath[i][1]]) >= 13:
				end_pos = i+1
				break
		if axis == 0:
			pos = [start_pos]
		else:
			pos = [end_pos]
		local_st = start_pos
		local_ed = end_pos
		while 1:
			temp_pos, comm_step = Trans.ldm_slice(newPath, indices, local_ed, local_st, axis, rma)
			pos.append(temp_pos)
			comm.append(comm_step)
			if axis == 0:
				local_st = temp_pos
			else:
				local_ed = temp_pos
			comm_len.append(len(comm_step))
			if local_st == local_ed:
				break
		if axis == 1:
			pos.reverse()
			comm.reverse()
			comm_len.reverse()
		return pos, comm, comm_len, start_pos, end_pos

	@staticmethod
	def gen_fused_map(newPath, indices, comm, st_pos, ed_pos):
		inmap = []
		outmap = []
		fused_map = []
		#gen inmap
		for i in range(len(indices[newPath[st_pos][1]])):
			if indices[newPath[st_pos][1]][i] in comm:
				inmap.append(i)
		for i in range(len(indices[newPath[ed_pos][1]])):
			if indices[newPath[ed_pos][1]][i] in comm:
				outmap.append(i)
		#print(inmap, outmap)
		for i in range(st_pos, ed_pos):
			indices_0 = list(copy.deepcopy(indices[newPath[i][0]]))
			indices_1 = list(copy.deepcopy(indices[newPath[i][1]]))
			if len(indices_0) < 13 and len(indices_1) < 13:
				continue
			#print("str : ", indices_0," , ", indices_1)
			#print("set : ", indices[newPath[i][0]], " , ", indices[newPath[i][1]])
			pos = 0
			trans_map_step = []
			left_map = []
			right_map = []
			for j in range(len(indices_1)-1, -1, -1):
				if indices_1[j] not in comm:
					right_map.append(pos)
					pos += 1
				else:
					indices_1.remove(indices_1[j])
			pos = 0
			for j in range(len(indices_0)-1, -1, -1):
				if indices_0[j] not in comm:
					left_map.append(pos)
					pos += 1
				else:
					indices_0.remove(indices_0[j])
			_comm = []
			trans_map_step = [left_map, right_map]
			for j in range(len(indices_0)):
				if indices_0[j] in indices_1:
					_comm.append([j, indices_1.index(indices_0[j])])
			#print("first: ", trans_map_step)
			#print("comm: ", _comm, "length : ", len(indices[newPath[i][1]]) - len(comm), len(indices[newPath[i][2]]) - len(comm), len(comm))
			#print(len(indices_1))
			trans_map_step_temp = []
			for elem in _comm:
				trans_map_step[0].remove(elem[0])
				trans_map_step[0].append(elem[0])
				trans_map_step[1].remove(elem[1])
				trans_map_step_temp.append(elem[1])
			trans_map_step[1] = trans_map_step_temp + trans_map_step[1]
			#print("end: ", trans_map_step)
			fused_map.append(trans_map_step)
		return fused_map, inmap, outmap

	@staticmethod
	def gen_fused_map_rma(newPath, indices, comm, st_pos, ed_pos):
		inmap = []
		outmap = []
		fused_map = []
		#gen inmap
		for i in range(len(indices[newPath[st_pos][1]])):
			if indices[newPath[st_pos][1]][i] in comm:
				inmap.append(i)
		for i in range(len(indices[newPath[ed_pos][1]])):
			if indices[newPath[ed_pos][1]][i] in comm:
				outmap.append(i)
		#print(inmap, outmap)
		for i in range(st_pos, ed_pos):
			indices_0 = list(copy.deepcopy(indices[newPath[i][0]]))
			indices_1 = list(copy.deepcopy(indices[newPath[i][1]]))
			if len(indices_0) < 13 and len(indices_1) < 13:
				continue
			#print("str : ", indices_0," , ", indices_1)
			#print("set : ", indices[newPath[i][0]], " , ", indices[newPath[i][1]])
			pos = 0
			trans_map_step = []
			left_map = []
			right_map = []
			for j in range(len(indices_1)-1, -1, -1):
				if indices_1[j] not in comm:
					right_map.append(pos)
					pos += 1
				else:
					indices_1.remove(indices_1[j])
			pos = 0
			for j in range(len(indices_0)-1, -1, -1):
				if indices_0[j] not in comm:
					left_map.append(pos)
					pos += 1
				else:
					indices_0.remove(indices_0[j])
			_comm = []
			trans_map_step = [left_map, right_map]
			for j in range(len(indices_0)):
				if indices_0[j] in indices_1:
					_comm.append([j, indices_1.index(indices_0[j])])
			#print("first: ", trans_map_step)
			#print("comm: ", _comm, "length : ", len(indices[newPath[i][1]]) - len(comm), len(comm))
			#print(len(indices_1))
			trans_map_step_temp = []
			for elem in _comm:
				trans_map_step[0].remove(elem[0])
				trans_map_step[0].append(elem[0])
				trans_map_step[1].remove(elem[1])
				trans_map_step_temp.append(elem[1])
			trans_map_step[1] = trans_map_step_temp + trans_map_step[1]
			#print("end: ", trans_map_step)
			fused_map.append(trans_map_step)
		return fused_map, inmap, outmap

	@staticmethod
	def gen_whole_fused_map(newPath, indices, comm, pos, rma=False):
		fused_map = []
		step_len = []
		for i in range(len(pos) - 1):
			if not rma:
				fused_map_step, inmap, outmap = Trans.gen_fused_map(newPath, indices, comm[i], pos[i], pos[i+1])
				fused_map.append([fused_map_step, inmap, outmap])
				fused_len = 0
				for step in fused_map_step:
					fused_len += len(step[0])
					fused_len += len(step[1])
				step_len.append(fused_len)
			else:
				fused_map_step, rma_map, inmap, outmap = Trans.gen_fused_map_rma(newPath, indices, comm[i], pos[i], pos[i+1])
			#print(fused_map_step, inmap, outmap, pos[i], pos[i+1])
		return fused_map, step_len

	@staticmethod	
	def path_contraction_cost(newPath, indices):
		cost = 0
		for step in newPath:
			dA = len(indices[step[0]])
			dB = len(indices[step[1]])
			dC = len(indices[step[2]])
			cost += 2**(int((dA + dB + dC)/2))
		return cost

	@staticmethod
	def find_critical_tensors(lifetime, target):
		critical = []
		for node in lifetime:
			if len(node[1]) == target:
				critical.append(copy.deepcopy(node))
		return critical

	@staticmethod
	def replace_candidate_from_criticals(critical):
		if len(critical) > 0:
			candidate = set() | critical[0][1]
			for i in range(1, len(critical)):
				temp = candidate & critical[i][1]
				candidate = temp
			return candidate
		return []

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
		opened = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,29,32,33,35,36,58,113]
		for step in newPath:
#			if (step[0] in opened) or (step[1] in opened):
#				print(step[0], step[1], step[2] - len(newPath))
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

def try_test():
	trans = Trans()
	test = Test()
	path_int, data_tensor = load_data("path.txt")
	newPath = trans.trans_path_index(path_int)
	test.test_new_path(newPath, True)
	indices = create_indices_set(newPath, data_tensor)
	test.test_indices(indices, newPath)
	children = trans.trans_path_to_tree(newPath)
	stem = trans.detect_stem(newPath, indices, 20)
	stem, newPath = trans.reverse_stem(stem, indices, newPath)
	trans.branch_merge(stem, indices, newPath, 0)
	test.test_indices(indices, newPath)
	finalPath = trans.stem_to_path(stem, newPath, indices)
	test.test_new_path(finalPath, True)
	path = trans.retrans_path_index(newPath)
