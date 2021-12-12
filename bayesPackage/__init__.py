import random
from bayesPackage.alphaCharClass import *
from bayesPackage.numericCharClass import *
import csv
from random import randint
import numpy as np

#import pandas as csv
###################################
# CS 470: Artifical Intelligence
# Author Jake Stuck
# License: MIT License 2021
################################


###############################################################
initialTraDat = "LetterRecognitioninitTrainingSet.txt"
moreTraDat = "optdigitsTrain.txt"
matureDat = "opticalDigitsTest.txt"
holisticList = []

###############################################################
###############################################################
# boolean
testFlg = True


# faking a pointer that I can access at the global level
# assigning the value one so I can override its place in memory
# the value one is also in place for multiplication protection
priorPointer = [1]



###############################################################

# Method: main()
# Process: Calls in all supporting functions
# Returns: None
def main():
	alphaSetFlg = False
	# Boolean flag for training status
	isTraining = True

	if testFlg == True:
		print("\n___________________________________")
		print("\nBayesian Handwritten Character analysis ")
		print("\n___________________________________")
		print("\n Loading in data text files")

	# reads in text data prior to being passed into a object
	holisticAList = loadInVectors(initialTraDat)
	holisticNList = loadInVectors(moreTraDat)

	if testFlg == True:
		print("\n___________________________________")
		print("\n Converting text data into a dataframe")
		print("\n___________________________________")
	## Alpha char set
	alphaObj = alphaCharClass(holisticAList, initialTraDat)
	alphaList = alphaObj.holisticAlphalList
	alphaFil = alphaObj.alphaCharSetFil
	alphaFrame = alphaObj.organizeAlphaSets( alphaFil )
	## Training set
	numericTObj = numericCharClass(holisticNList, moreTraDat, isTraining )
	numericTList = numericTObj.holisticNList
	numericTFil = numericTObj.numCharSetFil
	numericTFrame = numericTObj.organizeNumSets(numericTFil)
	## refined set
	numericRObj = numericCharClass(holisticNList, matureDat, isTraining)
	numericRList = numericRObj.holisticNList
	numericRFil = numericRObj.numCharSetFil
	numericRFrame = numericRObj.organizeNumSets(numericRFil)
	## Calling the classifier with numeric data
	# Train numeric call
	naiveBayes(numericTFrame, isTraining, alphaSetFlg, numericTObj)
	# Refined numeric call
	isTraining = False
	naiveBayes(numericRFrame, isTraining, alphaSetFlg, numericRObj)
	## Reset to true to avoid any scope issues
	isTraining = True
	#rescope to true prior to feeding alpha data into the classifier
	alphaSetFlg = True
	## feed alpha data into classifier
	#Train call
	isAlphaChar = True
	naiveBayes(alphaFrame, isTraining, isAlphaChar, alphaObj)

	isTraining = False
	#Refined call

	naiveBayes(alphaFrame, isTraining, isAlphaChar, alphaObj)


# Method: loadInVectors()
#
# Process: uses a file to load in the dataset and converts
# the dataset into a python list.
# Returns: A python list
#
def loadInVectors(fileN):
	with open(fileN, newline ='') as csvfile:
		delim = csv.reader(csvfile, delimiter=',', quotechar='|')
		testFlg = False
		for row in delim:
			testFlg = False
			if testFlg == True:
				print(",".join(row))
			holisticList = ",".join(row)
	return holisticList


# Method: naiveBayes
# Process: runs the naiveBayes classifer recursively
# and calls in any helper functions
#
# Returns: The likelyhood that an object given its state space coordinates in dataframe
# of vectors matches a particular handwritten character
def naiveBayes(dframe, isTraining, isAlphaChar, datasetObj):

	## return value, 0 as a place holder.
	probability = {}
	meanArr = []
	stdArr = []
	dataList = []

	# Splice random half of the dataset
	# If it is not a training set only return 20
	# rows to feed into classifier
	splicedData = selectRandomSec(dframe, isTraining)
	if testFlg == True:
		print("dataframe has been spliced")
	# Convert pandas dataframe to numpy array for
	# easier caclulation of covariance and Gaussian Multivariate
	print("\n____________________________________")
	print("\nConverting dataframe to numpy array prior to Multivariate Calculations")
	dataArr = splicedData.to_numpy()
	print("\n___________________________________")
	print("\n Displaying converted npArr")
	print(dataArr)
	# Splice random half of the dataset
	# If it is not a training set only return 20
	# rows to feed into classifier
	splicedData = selectRandomSec(dframe, isTraining)
	if testFlg == True:
		print("dataframe has been spliced")
	# Convert pandas dataframe to numpy array for
	# easier caclulation of covariance and Gaussian Multivariate
	print("\n____________________________________")
	print("\nConverting dataframe to numpy array prior to Multivariate Calculations")
	dataArr = splicedData.to_numpy()
	print("\n___________________________________")
	print("\n Displaying converted npArr")
	print("\n___________________________________")
	print(dataArr)
	# Calculate mean, covariance and gaussian probability in each
	# row of the np array, from there I can utilize that
	# to call in Np.Gauss() from numpy which can only take in
	# one argument at a time
	index = 0
	while index < dataArr.size -1 and index < 20:
		print(dataArr[index])
		print("\n_________________")
		print("\n Displaying integer Arr Values")
		# Since the alpha dataset contains strings, and
		# the first few columns of the numeric dataset contain
		dataList = (dataArr[index][1:])
		# if it is training then a dataset that is more
		# normalized around the bell curve would be best
		# to start with.
		if isTraining == True:
			meanArr = np.mean(dataList)
			if testFlg == True:
				print("\n_________________________")
				print("\ndisplaying mean arr:")
				print(meanArr)
				print("\n_________________________")

		# otherwise use standard deviation
		else:
			meanArr = np.std(dataList)

		# test output
		if testFlg == True:
			print("\n_________________________________")
			print("\n Displaying mean values:")
			print(meanArr)
			print("\n_________________________________")

		# Calculate covariance
		covariant = np.cov(dataList.astype(float))

		#test output
		if testFlg == True:
			print("\n_________________________________")
			print("\n Displaying covariant value")
			print(covariant)
			print("\n_________________________________")
		print("\n_________________")

		## Now that both the covariant and mean value have been computed
		## it is finally time to compute the gaussian distribution
		gaussVal = random.gauss(meanArr, covariant)

		#test output
		if testFlg == True:
			print("\n_________________________________")
			print("\nDisplaying Gaussian Value:")
			print(gaussVal)
			print("\n_________________________________")

		# Now that we have gaussian values that can be compared to the "Class" data
		# using the bayesian formula we can compute probabilities.

		# if it is not an alpha character
		# then use numericCharClass Obj var
		if isAlphaChar != True:
			return 5


		# Class 'class' data
		classD = datasetObj.alphaBayesClass

		for key, value in classD.items():
			value = classD[key]
			print("\n_________________________-")
			print("\ncalling bayesian probability function")
			print("\n________________________")

			probability[key] = determineP(value, gaussVal)

			# Class 'class' data

			#return probability

		# end of master loop
		index+=1

# Method:determineP
# Process: determines probability via the bayesian formula
# since one aspect of that formula is the prior class probability
# Returns calculated probability in a dictionary
def determineP(classD,data):

	likelyhood = classD * data
	postProb = priorPointer

	probability = likelyhood / classD
	postProb[0] = probability

	# ensuring that my "pointer" works
	assert priorPointer[0] == probability

	#Calculating the probability!
	bayesPosteriorProb = (classD * priorPointer[0])/ classD

	if testFlg == True:
		print("\n____________________________")
		print("\nDisplaying posterior prob value")
		print(bayesPosteriorProb)
		print("\n__________________________________")

	return bayesPosteriorProb



# Method: dispProbTable
# Process: helper function for
def dispProbTable():
	pass

# Method:  PrintInitDataSet
# Process: Gets the aggeragate and mean of each row
# Returns: None, printed aggeragate and mean of each row
def PrintInitDataSet(dframe):
	# Reference point for what exactly does a non numeric character look like?
	alphaAgg = dframe.agg

	# This dataframe holds the average of each row
	# The individual average of each row is an average of the
	# the vectors state space coordinates, these averages are what
	# can be used to distingush each handwritten character
	alphaAvg = dframe.mean(axis=0)

	# test output
	if testFlg == True:
		print("\n-------------------------")
		print("Start of init alpha dataset output")
		print("\n-------------------------")
		print(alphaAvg)
		print(alphaAgg)
		print("\n-------------------------")
		print("End of init alpha dataset output")
		print("\n-------------------------")


# Method: selectRandomSec()
# Process: Using a randomly generated value
# this method selects a section of the dataframe
# Returns: randomly scliced dataframe
def selectRandomSec(dframe, isTraining):
	dframeSize = dframe.size
	## alows for truly random_state sampling
	randomInteger = randint(0,9)
	## Splices dataframe in half using random sampling without
	## replacement
	returnDframe = dframe.sample(frac=0.5, replace=False,
								 random_state=randomInteger)
	# test output
	if testFlg == True:
		print("\n-------------------------")
		print("\n Begin output of spliced dataset")
		print("\n-------------------------")
		print(returnDframe)
		print("\n-------------------------")
		print("\n End output of spliced dataset")
		print("\n-------------------------")

	# Checks for training flag to propely determine proper mutation to the
	# dataframe
	if isTraining != True:
		## Sample a random twenty if the classifier is not training.
		newReturnDframe = returnDframe.sample(20,
											 replace = False)
		return newReturnDframe

	# Otherwise return as is
	return returnDframe



if __name__=='__main__':
	main()
