#Functions for extracting data

from PyPDF2 import PdfFileReader
from bs4 import BeautifulSoup
import urllib2
from docx import Document #python-docx pacakage
import os
import codecs
from pyPdf import PdfFileWriter, PdfFileReader
from StringIO import StringIO

################################
#global variables

path = "./DataExtractorTests/"

################################
#helper functions

def soupToText(soup):
	"""
	Args:
		BeautifulSoup soup object of an html file

	Returns:
		a string of the text stripped of the javascript
	"""

	# kill all script and style elements
	for script in soup(["script", "style"]):
	    script.extract()    # rip it out

	# get text
	text = soup.get_text()

	# break into lines and remove leading and trailing space on each
	lines = (line.strip() for line in text.splitlines())
	# break multi-headlines into a line each
	chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
	# drop blank lines
	text = '\n'.join(chunk for chunk in chunks if chunk)

	return text


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

def htmlToText(path):
	"""
	Args:
		path to html file

	Returns:
		a string of the html file
	"""
	with codecs.open(path, encoding='utf-8') as html:
		soup = BeautifulSoup(html)

	return soupToText(soup)


def urlHTMLToText(url):
	"""
	Args:
		url that corresponds to html

	Returns:
		a string of the text on the url
	"""
	doc = urllib2.urlopen(url).read()
	soup = BeautifulSoup(doc)

	return soupToText(soup)


def urlPDFToText(url):
	"""
	Args:
		url that corresponds to pdf

	Returns:
		a string of the text on the url
	"""
	doc = urllib2.urlopen(url).read()
	memoryFile = StringIO(doc)
	pdfFile = PdfFileReader(memoryFile)

	output = ""
	for page in PDF.pages:
	     output = output + page.extractText()

	return output

################################
#doc functions
#Good resource: http://www.konstantinkashin.com/blog/2013/09/25/scraping-pdfs-and-word-documents-with-python/

def docToText(path):
	"""
	Turns docx file into a string.

	Note: it concatenates paragraphs and does not 
	distinguish between beginning and end of paragraphs

	Args:
		path to docx file

	Returns:
		a string of the docx file
	"""
	doc = Document(path)

	output = ""
	for paragraph in doc.paragraphs:
		output = output + paragraph.text

	return output

def oldDocToTest(path):
	"""
	Turns doc file into a string.

	Args:
		path to doc file

	Returns:
		a string of the doc file
	"""
	name = path[:-4]
	textFile = name + ".txt"
	os.system("antiword " + path + " > " + textFile)

	with codecs.open(textFile, encoding='utf-8') as text:
		output = text.read()
	os.system("rm " + textFile)

	return output




if __name__ == "__main__":
	#Run tests
	print "pdf test: "
	print pdfToText(path + "test.pdf")

	print "urltest: "
	print urlToText('http://www.alec.org/model-legislation/72-hour-budget-review-act/')

	print "html test: "
	print htmlToText(path + "test.html")

	print "old doc test: "
	print oldDocToTest(path + 'test.doc')

