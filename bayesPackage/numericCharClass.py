import pandas

class numericCharClass:

	# member variables
	holisticNumList = []
	numCharSetFil = " "
	numDataframe = 0
	isTrainingSet = True
	numericBayesClass = {}

    # just a constructor
	def __init__(self, holisticNList, numCharSetFil, isTraining):
			self.holisticNList = holisticNList
			self.numCharSetFil = numCharSetFil
			self.isTrainingSet = isTraining
			#self.

    # Method: organizeNumSets
    # Process: reads in the numeric data set and
    # converts it into a pandas dataframe for easy data
    # interpretation and management
	def organizeNumSets(holisticNumList, numCharSetFil):

		table_obj = pandas.read_table(numCharSetFil, delimiter = ",")
		return table_obj
