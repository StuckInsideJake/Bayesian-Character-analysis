import csv
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
	organizeInitSets(holisticList)
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

def organizeInitSets(holisticList, Alist, Blist, Clist):

	if testFlg == True:
		print(holisticList)

	for char in holisticList:
		if char == "A":
			# should grab its index and the 18 following
			Alist = holisticList[char:18]
		if char == "B":
			Blist = holisticList[char:18]
		if char == "C":
			Clist = holisticList[char:18]
def averageLikeRows():
	pass

def naiveBayes():
	pass 






if __name__=='__main__':
	main()
