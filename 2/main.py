import pygraphml

from typing import Dict, List, Set, Tuple
import networkx
import matplotlib.pyplot as plt
from queue import Queue

DEFAULT_ENUMVAL: int = 1e12


class Node:
    def __init__(self, val: int):
        self.__neighbors: Set[Node] = set()
        self.val: str = val
        self.level: int = -1
        self.x: int = -1
        self.enumVal = DEFAULT_ENUMVAL

    def addNode(self, child):
        if child:
            self.__neighbors.add(child)
            child.__neighbors.add(self)

    def getNeighbors(self):
        return self.__neighbors

    def __ne__(self, other):
        return self.val != other.val

    def __eq__(self, other):
        return self.val == other.val

    def __hash__(self):
        return int(self.val[1:])

    def __lt__(self, other):
        selfEnumList: List[int] = self.__calculateEnumList()
        otherEnumList: List[int] = other.__calculateEnumList()
        if not selfEnumList:
            return True
        if not otherEnumList:
            return False
        for i in range(min(len(selfEnumList), len(otherEnumList)) - 1):
            if selfEnumList[i] != otherEnumList[i]:
                return selfEnumList[i] < otherEnumList[i]
        return len(selfEnumList) < len(otherEnumList)

    def __calculateEnumList(self) -> List:
        lst: List[int] = list()
        for node in self.__neighbors:
            lst.append(node.enumVal)

        lst.sort(reverse=True)
        return lst

    def getVal(self) -> int:
        return self.val


class Graph:
    def __init__(self, root: Node = None):
        self.root: Node = root
        self.nodes: Set[Node] = set()
        self.sortedEnumeratedNodeList : List[Node] = list()

    def parseGraphFromFile(self, graph_desctiption_filepath: str):
        parser = pygraphml.GraphMLParser()
        g = parser.parse(graph_desctiption_filepath)
        rawNodes = g.nodes()
        if not rawNodes:
            return
        for rawNode in rawNodes:
            self.__insertNode(None, rawNode)
        self.__startEnumerate()

    def __startEnumerate(self):
        first : Node = next(iter(self.nodes))
        first.enumVal = 0
        currEnumVal: int = [1]
        while currEnumVal[0] < len(self.nodes):
            self.__enumerate(currEnumVal)
        self.__levelEnumerate()
        pass

    def __levelEnumerate(self):
        q : Queue = Queue()
        maxlevel : int = 1
        for i in reversed(range(0, len(self.sortedEnumeratedNodeList))):
            q.put((maxlevel, self.sortedEnumeratedNodeList[i]))
            while not q.empty():
                r : Tuple[int, Node] = q.get()
                level : int = r[0]
                n : Node = r[1]
                if n.level != -1:
                    continue
                n.level = level
                maxlevel = max(maxlevel, level)
                for neighbor in n.getNeighbors():
                    q.put((level + 1, neighbor))

        

    def __enumerate(self, currEnumVal):
        countedEnumVals: List[Node] = list()
        for node in self.nodes:
            if node.enumVal == DEFAULT_ENUMVAL:
                countedEnumVals.append(node)
        bestNode = min(countedEnumVals)
        bestNode.enumVal = currEnumVal[0]
        self.sortedEnumeratedNodeList.append(bestNode)
        currEnumVal[0] += 1
        

    def __insertNode(self, parent: Node, rawNewNode: pygraphml.node.Node):
        newNode: Node = Node(rawNewNode.id)
        if newNode in self.nodes:
            return
        self.nodes.add(newNode)
        if parent:
            parent.addNode(newNode)

        for child in rawNewNode.children():
            self.__insertNode(newNode, child)

added : Set[Node] = set()

def addNode(gr: networkx.DiGraph, node: Node):
    if node in added:
        return
    added.add(node)
    nodeName: str = str(node.level) + " (" + str(node.enumVal) + ")"
    gr.add_node(nodeName, loc=(node.enumVal, node.level))
    for child in node.getNeighbors():
        childName: str = str(child.level) + " (" + str(child.enumVal) + ")"
        gr.add_edge(nodeName, childName)
        addNode(gr, child)


def printg(graph: Graph):
    gr = networkx.DiGraph()
    for node in graph.nodes:
        addNode(gr, node)
    networkx.draw(gr, networkx.get_node_attributes(
        gr, 'loc'), with_labels=False, node_size=10)
    plt.savefig('output.png')


t: Graph = Graph()
t.parseGraphFromFile("g27.xml")
printg(t)
pass
