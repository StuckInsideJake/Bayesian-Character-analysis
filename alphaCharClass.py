import pandas as csv
class alphaCharClass:

	# member variables
	holisticAlphalList = []
	alphaCharSetFil = " "

	alphaDataframe = 0


	def __init__(self, holisticAlphalList, alphaCharSetFil):
			self.holisticAlphalList = holisticAlphalList
			self.alphaCharSetFil = alphaCharSetFil

	def organizeAlphaSets(holisticList, alphaCharSetFil):

		table_obj = read_table(alphaCharSetFil, delimiter = ",")

		return table_obj
