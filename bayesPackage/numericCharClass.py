import pandas


class numericCharClass:

	# member variables
	holisticNlList = []
	numericCharSetFil = " "
	numericDataframe = 0

	# test flag
	testFlag = True

	# Bayes "Class" dictionary
	numericBayesClass = {}

	# Method: constructor
	# Process: Initialize all member variables
	# Returns: alphaCharClass as an object
	def __init__(self, holisticNlList, numericCharSetFil):

			self.holisticNlList = holisticNlList
			self.numericCharSetFil = numericCharSetFil
			self.alphaDataframe = self.organizeNumSets(numericCharSetFil)
			self.numericBayesClass = self.organizeClassData()


	# Method: organizeAlphaSets
	# Process: reads the inputted file from the constructor,
	# and converts it into a pandas dataframe
	# Returns: Pandas dataframe
	def organizeNumSets(self, alphaCharSetFil):

		# onvert to df
		table_obj = pandas.read_table(alphaCharSetFil, delimiter = ",")

		# assigns table_obj dataframe to the object reference for it
		self.alphaDataframe = table_obj

		return table_obj
	# Method: calculateDictMean
	# Process: Acts as a helper function for organizeClassData,
	# takes in a dictionary and returns a new dictionary with a alpha character
	# as a key and a numeric character representing the mean as the value
	# Returns: dict
	def calculateDictMean(self, dictionary):
		tempDict = {}
		localSw = self.testFlag

		for key, values in dictionary.items():

			print(key)
			print(values)
			tempDict[key] = self.calcListMean(values)

		localSw = True
		if localSw == True:
			print("\n________________________")
			print("\n Displaying mean dict values in NumericCharClass:")
			print(tempDict)
			print("\n______________________________")
		return tempDict

	# Method: calcListMean
	# Process: since python dictionaries are glorified hash tables
	# built on lists, I need to make the average on list types not dict types
	# Returns: average value of the list
	def calcListMean(self,values):
		return sum(values) / len(values)


	# Method: organizeClassData
	# Process: Acts as a helper function for the constructor
	# takes in alpha data as a basic python list, creates
	# and dumps the first column representing the
	# then it creates a dictionary with the mean associated with a particular character
	# Returns: mean alpha character dictionary
	def organizeClassData(self):

		localSw = self.testFlag
		localSw = True
		# Convert to dict
		dfDictl = self.alphaDataframe.to_dict("split")
		# Convert to list of lists using key

		numDict = {}
		returnDict = {}

		# Treat lists of lists as kind of like a 2d array
		# outer
		for key, value in dfDictl.items():
			# inner
			localSw = False
			if localSw == True:
				print("\n__________________")
				print("\n Displaying numeric key values")
				print(key)
				print("\n__________________")
				print("\n________________________")
				print("\n Printing numeric values")
				print(value)
				print("\n________________________")

		if localSw == True:
				print("\n_______________________")
				print("Displaying state space coords in a dict from numericCharClass")
				print(numDict)
				print("\n_______________________")

		returnDict = self.calculateDictMean(numDict)
		return returnDict


