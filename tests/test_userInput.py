import sys

sys.path.append('../')
import projSplit as ps
import lossProcessors as lp
import pytest

def createNewTests(argsInOrder,names,trials,expectedMtx,Processor):
    newTests = []
    for i in range(len(argsInOrder)):
        name = names[i]
        for j in range(len(trials)):
            args = argsInOrder.copy()
            args[i] = trials[j]
            expect = expectedMtx[i][j]
            newTests.append((Processor, args, name, expect))
    return newTests

AllTests = []


Processor = lp.Forward2Backtrack
argsInOrder = [1.0,1.0,0.7,1.1,None]
names = ["step","Delta","decFactor","growFactor","growFreq"]
trials = [0.0,1.0,1.5,0.5,-1.0,"howdy"]
expectedMtx = [[1.0,1.0,1.5,0.5,1.0,1.0],
               [1.0,1.0,1.5,0.5,1.0,1.0],
               [0.7,0.7,0.7,0.5,0.7,0.7],
               [1.0,1.0,1.5,1.0,1.0,1.0],
               [ 10,  1,  1, 10, 10, 10]
              ]

AllTests.extend(createNewTests(argsInOrder,names,trials,expectedMtx,Processor))

Processor = lp.Forward2Fixed
argsInOrder = [1.0]
names = ["step"]
trials = [0.0,1.0,1.5,0.5,-1.0,"howdy"]
expectedMtx = [[1.0,1.0,1.5,0.5,1.0,1.0]
              ]
AllTests.extend(createNewTests(argsInOrder,names,trials,expectedMtx,Processor))

Processor = lp.Forward2Affine

argsInOrder = [1.0]
names = ["Delta"]
trials = [0.0,1.0,1.5,0.5,-1.0,"howdy"]
expectedMtx = [[1.0,1.0,1.5,0.5,1.0,1.0]
              ]
AllTests.extend(createNewTests(argsInOrder,names,trials,expectedMtx,Processor))

Processor = lp.Forward1Fixed
argsInOrder = [1.0,0.1]
names = ["step","alpha"]
trials =       [0.0,1.0,1.5,0.5,-1.0,"howdy"]
expectedMtx = [[1.0,1.0,1.5,0.5, 1.0,    1.0],
               [0.1,0.1,0.1,0.5, 0.1,    0.1]
              ]
AllTests.extend(createNewTests(argsInOrder,names,trials,expectedMtx,Processor))

Processor = lp.Forward1Backtrack
argsInOrder = [1.0,0.1,0.7,1.0,None]
names = ["step","alpha","delta","growFac","growFreq"]

trials =       [0.0,1.0,1.5,0.5,-1.0,"howdy"]
expectedMtx = [[1.0,1.0,1.5,0.5, 1.0,    1.0],
               [0.1,0.1,0.1,0.5, 0.1,    0.1],
               [0.7,0.7,0.7,0.5, 0.7,    0.7],
               [1.0,1.0,1.5,1.0, 1.0,    1.0],
               [10 ,  1,  1, 10,  10,     10]
              ]
AllTests.extend(createNewTests(argsInOrder,names,trials,expectedMtx,Processor))

Processor = lp.BackwardExact
argsInOrder = [1.0]
names = ["step"]
trials = [0.0,1.0,1.5,0.5,-1.0,"howdy"]
expectedMtx = [[1.0,1.0,1.5,0.5,1.0,1.0]
              ]
AllTests.extend(createNewTests(argsInOrder,names,trials,expectedMtx,Processor))

Processor = lp.BackwardCG

argsInOrder = [0.9,1.0,100]
names = ["sigma","step","maxIter"]

trials =       [0.0,1.0,1.5,0.5,-1.0,"howdy"]

expectedMtx = [[0.0,0.9,0.9,0.5, 0.9,    0.9],
               [1.0,1.0,1.5,0.5, 1.0,    1.0],
               [100,  1,  1,100, 100,    100]
              ]
AllTests.extend(createNewTests(argsInOrder,names,trials,expectedMtx,Processor))

Processor = lp.BackwardLBFGS

argsInOrder = [1.0,0.9,10,0.0001,0.9,0.7,1.1,100,20]

names = ["step","sigma","m","c1","c2","shrinkFactor","growFactor","maxiter","lineSearchIter"]

trials =       [0.0,1.0,1.5,0.5,-1.0,"howdy"]

expectedMtx = [[1.0,1.0,1.5,0.5, 1.0,    1.0],
               [0.0,0.9,0.9,0.5, 0.9,    0.9],
               [10,   1,  1, 10,   10,     10],
               [1e-4,1e-4,1e-4,0.5,1e-4,1e-4],
               [0.9,0.9,0.9,0.5,0.9,0.9],
               [0.7,0.7,0.7,0.5,0.7,0.7],
               [1.1,1.1,1.5,1.1,1.1,1.1],
               [100,1,1,100,100,100],
               [20,1,1,20,20,20]
              ]

AllTests.extend(createNewTests(argsInOrder,names,trials,expectedMtx,Processor))

@pytest.mark.parametrize("Processor,args,testAttribute,expected",AllTests)
def test_incorrect(Processor,args,testAttribute,expected):
    processObj = Processor(*args)
    attribute = getattr(processObj,testAttribute)
    assert attribute == expected



