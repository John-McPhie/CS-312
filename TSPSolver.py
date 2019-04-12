#!/usr/bin/python3
from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import random
import math
import system

def reduce(m):
	#Copy Matrix
	matrix = m.copy()
	reduction_cost = 0

	#Subtract the min value in each row from every element in that row
	for row in matrix:
		min = row.min()
		if min != 0 and min != float('inf'): #Zero already in row, or all inf
			row[:] = row[:] - min
			reduction_cost += min

	for col in matrix.T:
		min = col.min()
		if min != 0 and min != float('inf'): #col already has zero, or all inf
			col[:] = col[:] - min
			reduction_cost += min

	return matrix, reduction_cost


class Node:
	def __init__(self, parent, path, matrix, remaining = set()):
		#Initialize Variables
		self.parent = parent
		self.path = path
		self.children = None

		if self.parent is not None: #Non-root node.
			#Get LB: parent_LB + transition_cost + reduction_cost
			from_city = self.path[-2]
			to_city = self.path[-1]
			transition_cost = self.parent.matrix[from_city._index][to_city._index]
			self.matrix, reduce_cost = reduce(matrix)
			self.LB = self.parent.LB + transition_cost + reduce_cost

			#Get a list of cities not visited
			self.remaining = self.parent.remaining.copy()
			self.remaining.remove(self.path[-1])
		else: #Root Node
			#Initialize Variables
			self.matrix, self.reduce_cost = reduce(matrix)
			self.LB = self.reduce_cost
			self.remaining = remaining


	def expand(self): #Creates and returns the number of children
		if self.children is not None: #Children already created, skip work
			return 0 #No new children were created
		else:
			self.children = []
			for city in self.remaining: #Create a new node for each remaining city
				m = self.matrix.copy()

				#Make row (from city), and col (to city) all infinity
				m[self.path[-1]._index,:] = float('inf')
				m[:,city._index] = float('inf')

				#You don't wan't to go back to that node, example A->B: [B,A]=INF
				m[city._index, self.path[-1]._index] = float('inf')
				self.children.append(Node(self,self.path + [city],m))
		return len(self.children)

	def print_path(self):
		path = []
		for city in self.path:
			path.append(city._name)
		print(path)

	def print_remaining(self): #Prints the remaining cities to visit
		remaining = []
		for city in self.remaining:
			remaining.append(city._name)
		print(remaining)

	#If a node is deeper than the other, then len(self.path) - len(other.path)
	#will be positive because it has a larger path (greater depth). We want
	#to give higher priority to deeper nodes, so we will boost it by decreasing
	#its LB by 3 * the increase in depth (making it more likely the min heap will
	#choose it).
	def __lt__(self, other): #Impliments the < operator for the priority queue
		depth_diff = len(self.path) - len(other.path)#depth separating nodes
		return (self.LB - 3 * depth_diff < other.LB) #give priority to deeper nodes

class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''
	def greedy( self,time_allowance=60.0 ):
		cities = self._scenario.getCities()
		start_time = time.time()
		s = self.greedy_fancy(cities[random.randint(0, len(cities)-1)])
		initial = s
		while s.cost == float('inf'):
			s = self.greedy_fancy(cities[random.randint(0, len(cities)-1)])

		end_time = time.time()
		results = {}
		results['cost'] = s.cost
		results['time'] = end_time - start_time
		results['count'] = 0
		results['soln'] = s
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		system.notify("Done", "Greedy")
		return results


	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''
	def branchAndBound( self, time_allowance=60.0 ):
		#Initialize Variables
		solution = None
		num_solutions = 0
		num_states = 0
		pruned = 0
		max_q = 0
		cities = self._scenario.getCities()
		start_time = time.time()
		default = self.defaultRandomTour()
		BSSF = default['cost']

		#Initialize Cost Matrix
		num_cities = len(cities)
		root_matrix = np.zeros((num_cities,num_cities))
		remaining = []
		for row in range(num_cities):
			for col in range(num_cities):
				root_matrix[row][col] = cities[row].costTo(cities[col])

		#Create Root and Priority Queue
		root = Node(None, [cities[0]], root_matrix, set(cities[1:]))
		nodes = [root]
		heapq.heapify(nodes)

		#Run Branch & Bound Algorithm
		while len(nodes) != 0 and time.time()-start_time < time_allowance:
			if len(nodes) > max_q: #Update max size of queue
				max_q = len(nodes)

			node = heapq.heappop(nodes) #Get new node from priority queue
			num_states += node.expand()
			for child in node.children:
				if time.time()-start_time >= time_allowance:
					break;

				if child.LB > BSSF: #BSSF is better, don't put into queue
					pruned += 1
				elif len(child.remaining) == 0: #complete solution
					first = child.path[0]._index
					last = child.path[-1]._index
					final = child.matrix[last][first]
					num_solutions += 1
					BSSF = child.LB + final
					solution = child
				else: #Put in queue
					heapq.heappush(nodes, child)

		end_time = time.time()
		if solution is None:#Initial BSSF was the best.
			system.notify("Timed Out")
			return default
		else:#Display results
			bssf = TSPSolution(solution.path)
			results = {}
			results['cost'] = bssf.cost
			results['time'] = end_time - start_time
			results['count'] = num_solutions
			results['soln'] = bssf
			results['max'] = max_q
			results['total'] = num_states
			results['pruned'] = pruned + len(nodes) #len(nodes): not dequeued
			system.notify("Done", "Branch and Bound")
			return results

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''
	def fancy( self,time_allowance=60.0 ):
		random.seed(time.time())
		init = 100000000
		T = init
		cities = self._scenario.getCities()
		final_index = len(cities) - 1
		num_solutions = 0
		greedies = set()

		start_time = time.time()
		for city in cities: #Get a greedy solution starting from each city
			greedies.add(self.greedy_fancy(city))

		s = next(iter(greedies)) #get first one
		bssf = s

		for solution in greedies: #Run simulated annealing on each greedy solution
			if solution.cost < bssf.cost:
				num_solutions += 1
				bssf = solution

			while T > 0 and time.time() - start_time < time_allowance:
				route = solution.route.copy()
				#randomly choose a solution in the neighborhood.
				index1 = random.randint(0, final_index)
				index2 = random.randint(0, final_index)
				temp = route[index1]
				route[index1] = route[index2]
				route[index2] = temp
				neighbor = TSPSolution(route)

				if neighbor.cost != float('inf'):
					num_solutions += 1
					if neighbor.cost < bssf.cost:
						bssf = neighbor
						T = init
					diff = neighbor.cost - s.cost
					if diff < 0:
						s = neighbor
					else:
						probability = math.exp(-diff/T)
						if probability > random.random():
							s = neighbor
					T = .99 * T - .000001 #I just experimented with this until is got better...

		end_time = time.time()
		system.notify("Done", "Fancy")
		results = {}
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = num_solutions
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	def greedy_fancy(self, city, time_allowance=60.0):
		cities = self._scenario.getCities()
		results = {}
		curr_city = city
		route = [curr_city]
		free_cities = set(cities)
		free_cities.remove(curr_city)
		while free_cities:
			next_city = min(free_cities, key=lambda x: curr_city.costTo(x))
			free_cities.remove(next_city)
			route.append(next_city)
			curr_city = next_city

		return TSPSolution(route)
