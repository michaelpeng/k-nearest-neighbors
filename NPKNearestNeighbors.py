import math
import operator
import random
import numpy as np

trainingSet = []
trainingLabels = []
compareSet = []
compareLabels = []

trainingdict = {}
comparedict = {}

def putIntoTrainingSet():
	global trainingSet
	with open('trainFeatures.csv') as feats:
		lines = feats.read().split("\n")
		trainingSet = [[float(num) for num in line.split(',') if num] for line in lines]

def putIntoTrainingLabels():
	global trainingLabels
	with open('trainLabels.csv') as feats:
		lines = feats.read().split("\n")
		trainingLabels = [[float(num) for num in line.split(',') if num] for line in lines]

def putIntocompareSet():
	global compareSet
	with open('valFeatures.csv') as feats:
		lines = feats.read().split("\n")
		compareSet = [[float(num) for num in line.split(',') if num] for line in lines]

def putIntocompareLabels():
	global compareLabels
	with open('valLabels.csv') as feats:
		lines = feats.read().split("\n")
		compareLabels = [[float(num) for num in line.split(',') if num] for line in lines]	

def mapSetsandLabels():
	global trainingdict
	global comparedict
	i = 0
	for key in trainingSet:
		trainingdict[str(key)] = trainingLabels[i]
		i+=1
	j = 0
	for key in compareSet:
		comparedict[str(key)] = compareLabels[j]
		j+=1

def getNeighbors(trainingSet, testInst, k):
	dists = []

	for i in range(len(trainingSet)):
		a = np.array(testInst)
		b = np.array(trainingSet[i])
		
		dist = np.linalg.norm(a-b)
		dists.append((trainingSet[i], dist))

	dists.sort(key=operator.itemgetter(1))
	neighbors = []
	for i in range(k):
		neighbors.append(dists[i][0])
	return neighbors

def getResponse(neighbors):
	votes = {}
	for i in range(len(neighbors)):
		reply = str(trainingdict[str(neighbors[i])])
		if reply in votes:
			votes[reply] += 1
		else:
			votes[reply] = 1
	sortvotes = sorted(votes.items(), key = operator.itemgetter(1), reverse=True)
	checklist = [float(sortvotes[0][0].replace("[", "").replace("]", ""))]
	i = 0
	j = 1
	while j < len(sortvotes) and sortvotes[i][0] == sortvotes [j][0]:
		checklist.append(float(sortvotes[j][0].replace("[", "").replace("]", "")))
		i += 1
		j += 1
	rand = random.randint(0, len(checklist)-1)
	return checklist[rand]


def accuracyTest(compareLabels, predict):
	correct = 0
	for i in range(len(compareLabels)):
		if compareLabels[i][0] == predict[i]:
			correct += 1
	return (correct/float(len(compareLabels)))*100.0


def main():
	putIntoTrainingSet()
	putIntocompareSet()
	putIntocompareLabels()
	putIntoTrainingLabels()
	mapSetsandLabels()

	predict = []
	k = 25
	for i in range(len(compareSet)):
		neighbors = getNeighbors(trainingSet, compareSet[i], k)
		result = getResponse(neighbors)
		predict.append(result)
		print('> predicted=' + repr(result))
	accuracy = accuracyTest(compareLabels, predict)
	print('Accuracy: ' + repr(accuracy) + '%')

main()