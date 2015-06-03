#Functions for extracting data

from PyPDF2 import PdfFileReader
from bs4 import BeautifulSoup

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

################################
#html functions

test_url_path = "/Users/jkatzsamuels/Desktop/dssg/sunlight/test_code/test.html"

def urlToText(path):
	"""
	Args:
		path to html file

	Returns:
		a string of the html file
	"""
	with open(path,'r') as html:
		soup = BeautifulSoup(html)
	output = soup.get_text()

	return output


