import csv
from Bayesian-Character-analysis import alphaCharClass
#import pandas as csv
###################################
# CS 470: Artifical Intelligence  				  #
# Author Jake Stuck
# License: MIT License 2021(C)
# Jake Stuck- All Rights reserved
###################################

initialTraDat = "LetterRecognitioninitTrainingSet.txt"
moreTraDat = "optdigitsTrain.txt"
matureDat = "opticalDigitsTest.txt"
holisticList = []

## initially training the classifier using three lists to
## recognize three sets of letters.
Alist = []
Blist = []
Clist = []

testFlg = True

#Calls in all supporting functions
def main():
	holisticList = loadInVectors(initialTraDat)

	alphaObj = alphaCharClass(holisticList, initialTraDat)
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


def naiveBayes():
	pass






if __name__=='__main__':
	main()
