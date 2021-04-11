import pygraphml

from typing import Dict, List, Set
import networkx
import matplotlib.pyplot as plt

class Node:
	def __init__(self, val : int):
		self.__children : Set[Node] = set()
		self.val : str = val
		self.depth : int = -1
		self.x : int = -1
	

	def addChild(self, child):
		child.depth = self.depth + 1
		self.__children.add(child)
		assert(len(self.__children) <= 2) # Implementation is not totaly ready for n-ary tree


	def getChildren(self):
		return self.__children

	def __ne__(self, other):
		return self.val != other.val


	def __eq__(self, other):
		return self.val == other.val


	def __hash__(self):
		return int(self.x)


	def enumerateNode(self, num):
		for i, child in enumerate(self.__children):
			child.enumerateNode(num)
			if(i == 0):
				self.x = num[0]
				num[0] += 1
		
		if not self.__children:
			self.x = num[0]
			num[0] += 1
	

	def shift(self):
		childList : list = list(self.__children)
		for i in range(1, len(childList)):
			shift = self.__getShift(childList[i-1].__getRightX(), childList[i].__getLeftX())
			childList[i].__makeShift(-shift)

	def __makeShift(self, shift : int):
		for child in self.__children:
			child.__makeShift(shift)
		self.x += shift


	def __getShift(self, maxLeft : int, minRight : int) -> int:
		return minRight - maxLeft + 2


	def __getLeftX(self) -> int:
		if self.__children:
			return list(self.__children)[0].__getLeftX()
		else:
			return self.x


	def __getRightX(self) -> int:
		if len(self.__children) > 1:
			return list(self.__children)[-1].__getRightX()
		else:
			return self.x

	def getVal(self) -> int:
		return self.val


class Tree:
	def __init__(self, root : Node = None):
		self.root : Node = root


	def parseGraphFromFile(self, graph_desctiption_filepath : str):
		parser = pygraphml.GraphMLParser()
		g = parser.parse(graph_desctiption_filepath)
		rawRoot : pygraphml.node.Node = None
		for node in g.nodes():
			if not node.parent():
				rawRoot = node
		
		if not rawRoot:
			return

		root : Node = Node(rawRoot.id)
		root.depth = 1
		self.root = root

		for node in rawRoot.children():
			self.__insertNode(self.root, node)

		self.__enumerate(root)
		self.root.shift()
	
	def __enumerate(self, node : Node):
		i = [0]
		node.enumerateNode(i)


	def __insertNode(self, parent : Node, rawNewNode : pygraphml.node.Node):
		newNode : Node = Node(rawNewNode.id)
		parent.addChild(newNode)

		for child in rawNewNode.children():
			self.__insertNode(newNode, child)


def addNode(gr : networkx.DiGraph, node : Node):
	nodeName : str = node.getVal() + " (" + str(node.x) + ")"
	gr.add_node(nodeName, loc = (node.x, -node.depth))
	for child in node.getChildren():
		childName : str = child.getVal() + " (" + str(child.x) + ")"
		gr.add_edge(nodeName, childName)
		addNode(gr, child)


def printg(tree : Tree):
	gr = networkx.DiGraph()
	addNode(gr, tree.root)
	networkx.draw(gr, networkx.get_node_attributes(gr, 'loc'), with_labels=False, node_size=10)
	plt.savefig('output.png')



t : Tree = Tree()
t.parseGraphFromFile("72tree.xml")
printg(t)
pass