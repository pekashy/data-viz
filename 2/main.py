import pygraphml

from typing import Dict, List, Set, Tuple
import networkx
import matplotlib.pyplot as plt
from queue import Queue
from statistics import median

DEFAULT_ENUMVAL: int = 1e12
W : int = 3

class Node:
    def __init__(self, val: int):
        self.__neighbors: Set[Node] = set()
        self.val: str = val
        self.level: int = -1
        self.enumVal : float = DEFAULT_ENUMVAL

    def addNode(self, child):
        if child:
            self.__neighbors.add(child)
            child.__neighbors.add(self)

    def getNeighbors(self):
        return self.__neighbors        

    def getChildren(self) -> list:
        children : list = list()
        for node in self.__neighbors:
            if(node.level > self.level):
                children.append(node)
        return children


    def getDummyNodesGenerated(self, sameLevelNode):
        if self.level != sameLevelNode.level:
            return 0
        bNodeInArgPlacedOnTheLeft : bool = sameLevelNode.enumVal <= self.enumVal
        
        argNodeChildren : list = sameLevelNode.getChildren()
        currentNodeChildren : list = self.getChildren()

        dummyNodesGenerated : int = 0

        for child in currentNodeChildren:
            childX : int = child.enumVal

            for argChild in argNodeChildren:
                argChildX : int = argChild.enumVal

                if bNodeInArgPlacedOnTheLeft:
                    if argChildX >= childX:
                        dummyNodesGenerated += 1
                else: # on the right
                    if argChildX <= childX:
                        dummyNodesGenerated += 1

        return dummyNodesGenerated

    def getDummyNodeTotalNumber(self, newCoord, sameLevelNodes : list):
        dummyNodesNum = 0
        oldCoord = self.enumVal
        self.enumVal = newCoord
        for node in sameLevelNodes:
            if self.enumVal == node.enumVal:
                continue
            dummyNodesNum += self.getDummyNodesGenerated(node) + abs(self.level - node.level) - 1
        self.enumVal = oldCoord
        return dummyNodesNum


    def dummyToggle(self, takenKoords : set, sameLevelNodes : list):
        takenKoords.remove(self.enumVal)
        bestIResults = (self.enumVal, self.getDummyNodeTotalNumber(self.enumVal, sameLevelNodes))

        for i in range(-W*10, W*10):
            if self.enumVal + i in takenKoords:
                continue
            res : int = self.getDummyNodeTotalNumber(self.enumVal + i, sameLevelNodes)
            if bestIResults[1] < res:
                if bestIResults[0] in takenKoords:
                    takenKoords.remove(bestIResults[0])
                bestIResults = (self.enumVal + i, res)
                takenKoords.add(bestIResults[0])

            if bestIResults[1] == res and abs(self.enumVal + i) < abs(bestIResults[0]):
                if bestIResults[0] in takenKoords:
                    takenKoords.remove(bestIResults[0])
                bestIResults = (self.enumVal + i, res)
                takenKoords.add(bestIResults[0])

        self.enumVal = bestIResults[0]
        takenKoords.add(self.enumVal)


    def toggle(self, visited : set, takenKoords : set):
        if self in visited:
            return
        visited.add(self)
        neighVals : List[int] = list()
        for node in self.__neighbors:
            neighVals.append(node.enumVal)
        
        if not neighVals:
            return

        takenKoords.add((self.level, self.enumVal))
        bestKoord : int = int(median(neighVals))
        i : int = 0
        # prevent collisions
        while (self.level, bestKoord + i) in takenKoords and (self.level, bestKoord - i) in takenKoords:
            i+=1
        
        # choose empty one
        finalBestCoord = bestKoord + i
        if (self.level, bestKoord + i) in takenKoords:
            finalBestCoord = bestKoord - i
        self.enumVal = finalBestCoord
        takenKoords.add((self.level, self.enumVal))

        for node in self.__neighbors:
            node.toggle(visited, takenKoords)


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
        self.levels : Dict[int, List[Node]] = dict()

    def parseGraphFromFile(self, graph_desctiption_filepath: str):
        parser = pygraphml.GraphMLParser()
        g = parser.parse(graph_desctiption_filepath)
        rawNodes = g.nodes()
        if not rawNodes:
            return
        for rawNode in rawNodes:
            self.__insertNode(None, rawNode)
        self.__startEnumerate()
        for i in range(100):
            toggled : Set[Node] = set()
            takenKoords : set = set()
            for node in self.nodes:
                node.toggle(toggled, takenKoords)
        
        for i in range(10):
            for level in self.levels:
                takenKoords = set()
                for node in self.levels[level]:
                    takenKoords.add(node.enumVal)
                for node in self.levels[level]:
                    node.dummyToggle(takenKoords, self.levels[node.level])
        
        
        for i in range(100):
            toggled : Set[Node] = set()
            takenKoords : set = set()
            for node in self.nodes:
                node.toggle(toggled, takenKoords)


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
        levelVacants : Dict[int : int] = dict()
        for i in reversed(range(0, len(self.sortedEnumeratedNodeList))):
            q.put((maxlevel, self.sortedEnumeratedNodeList[i]))
            while not q.empty():
                r : Tuple[int, Node] = q.get()
                level : int = r[0]
                n : Node = r[1]
                if n.level != -1:
                    continue
                if level not in levelVacants:
                    levelVacants[level] = W
                    self.levels[level] : List[Node] = list()

                if levelVacants[level] == 0:
                    q.put((level + 1, n))
                    continue
                levelVacants[level] -= 1
                n.level = level
                self.levels[level].append(n)
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
        if parent:
            parent.addNode(newNode)
            self.nodes.add(newNode)
            self.nodes.add(parent)
        
        if not rawNewNode.children():
            self.nodes.add(newNode)

        for child in rawNewNode.children():
            self.__insertNode(newNode, child)
    
    

added : Set[Node] = set()

def addNode(gr: networkx.DiGraph, node: Node):
    if node in added:
        return
    added.add(node)
    nodeName: str = node.getVal()
    gr.add_node(nodeName, loc=(node.enumVal, node.level))
    for child in node.getNeighbors():
        childName: str = child.getVal()
        gr.add_edge(nodeName, childName)
        addNode(gr, child)


def printg(graph: Graph):
    gr = networkx.DiGraph()
    for node in graph.nodes:
        addNode(gr, node)
    networkx.draw(gr, networkx.get_node_attributes(
        gr, 'loc'), with_labels=False, node_size=20)
    plt.savefig('output.png')


t: Graph = Graph()
t.parseGraphFromFile("g27.xml")
printg(t)
pass
