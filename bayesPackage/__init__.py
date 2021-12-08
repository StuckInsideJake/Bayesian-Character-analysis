import csv
from bayesPackage.alphaCharClass import *

#import pandas as csv
###################################
# CS 470: Artifical Intelligence  				  #
# Author Jake Stuck
# License: MIT License 2021(C)
# Jake Stuck- All Rights reserved
################################   ###

initialTraDat = "LetterRecognitioninitTrainingSet.txt"
moreTraDat = "optdigitsTrain.txt"
matureDat = "opticalDigitsTest.txt"
holisticList = []



testFlg = False

#Calls in all supporting functions
def main():

	holisticList = loadInVectors(initialTraDat)
	alphaObj = alphaCharClass(holisticList, initialTraDat)
	alphaList = alphaObj.holisticAlphalList

	alphaFil = alphaObj.alphaCharSetFil
	alphaFrame = alphaObj.organizeAlphaSets( alphaFil )

	# Reference point for what exactly does a non numeric character look like?
	alphaAgg = alphaFrame.agg

	# This dataframe holds the average of each
	alphaAvg = alphaFrame.mean(axis=1)

	print(alphaAvg)

	#organizeAlphaSets(holisticList, Alist, Blist, Clist)

#
#
#
def loadInVectors(fileN):
	with open(fileN, newline ='') as csvfile:
		delim = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in delim:
			if testFlg == True:
				print(",".join(row))
			holisticList = ",".join(row)
	return holisticList


def averageLikeRows(inList):
	if testFlg == True:
		print(inList)

#
#
#
def naiveBayes(dframe):
	pass




if __name__=='__main__':
	main()
