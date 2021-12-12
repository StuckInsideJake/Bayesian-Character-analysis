import pandas
import statistics

class alphaCharClass:

	# member variables
	holisticAlphalList = []
	alphaCharSetFil = " "
	alphaDataframe = 0

	# test flag
	testFlag = True

	# Bayes "Class" dictionary
	alphaBayesClass = {}

	# Method: constructor
	# Process: Initialize all member variables
	# Returns: alphaCharClass as an object
	def __init__(self, holisticAlphalList, alphaCharSetFil):

			self.holisticAlphalList = holisticAlphalList
			self.alphaCharSetFil = alphaCharSetFil
			self.alphaDataframe = self.organizeAlphaSets(alphaCharSetFil)

			self.alphaBayesClass = self.organizeClassData()


	# Method: organizeAlphaSets
	# Process: reads the inputted file from the constructor,
	# and converts it into a pandas dataframe
	# Returns: Pandas dataframe
	def organizeAlphaSets(self, alphaCharSetFil):

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
			tempDict[key] = self.calcListMean(values)

		localSw = False
		if localSw == True:
			print("\n________________________")
			print("\n Displaying mean dict values")
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
		localSw = False
		# Convert to dict
		dfDictl = self.alphaDataframe.to_dict("split")
		# Convert to list of lists using key
		dfDictData = dfDictl.get("data")
		alphabetDict = {}
		returnDict = {}

		# Treat lists of lists as kind of like a 2d array
		# outer
		for item in dfDictData:
			# inner
			for index, element in enumerate(item):
				# if element is alpha then
				# take the next few indicies after to compute a bayes "class" mean
				# of that letter
				if isinstance(element,str) == True:
					# For the most part there are 16 integers or coordinates
					# after a letter
					alphabetDict[element] = item[index+1:15]


				if localSw == True:
					print("\n_______________________")
					print("Displaying state space coords in a dict")
					print(alphabetDict)
					print("\n_______________________")
				returnDict = self.calculateDictMean(alphabetDict)
		return returnDict




