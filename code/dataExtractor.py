#Functions for extracting data

from PyPDF2 import PdfFileReader


################################
#pdf functions

def pdfToText(path):
	"""
	Args:
		path to pdf file

	Returns:
		a string of the pdf
	"""

	PDF = PdfFileReader(file(path, 'rb'))

	output = ""
	for page in PDF.pages:
	     output = output + page.extractText()

	return output


