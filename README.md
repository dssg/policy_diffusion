# Legislative Influence Detector

Legislators often lack the time to write bills, so they tend to rely on outside groups to help. Researchers and concerned citizens would like to know who’s writing legislative bills, but trying to read those bills, let alone trace their source, is tedious and time consuming. This is especially true at the state and local levels, where arguably more important policy decisions are made every day.

This project provides tools to help analyze and access government bills. Using the Sunlight Foundation’s collection of state bills and model legislation scraped from lobbying groups from around the country, we built tools to shed light on the origination and diffusion of policy ideas around the country, the effectiveness of various lobbying organizations, and the democratic nature of individual bills, all in near real time.

# How does it work?

We use the Smith-Waterman local-alignment algorithm to find matching text across documents. This algorithm grabs pieces of text from each document and compares each word, adding points for matches and subtracting points for mismatches. Unfortunately, the local-alignment algorithm is too slow for large sets of text, such as ours. It could take the algorithm thousands of years to finish analyzing the legislation. We improved the speed of the analysis by first limiting the number of documents that need to be compared. Elasticsearch, our database of choice for this project, efficiently calculates Lucene scores. When we use LID to search for a document, it quickly compares our document against all others and grabs the 100 most similar documents as measured by their Lucene scores. Then we run the local-alignment algorithm on those 100.

# Important Files

* text_alignment.py: contains our fast implementation of the smith-waterman algorithm.

## Environmental Variables
* POLICY_DIFFUSION
* LOGFILE_DIRECTORY: should not exist inside repository, to prevent repository bloating
* TEMPFILE_DIRECTORY: stores files created temporarily while the algorithm runs
* ELASTICSEARCH_IP
 
