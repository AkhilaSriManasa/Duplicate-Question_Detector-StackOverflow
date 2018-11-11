import pandas as pd
import numpy
import scipy
import sklearn
import csv
#connecting to sqlite first to get the database
import sqlite3
conn=sqlite3.connect('D:\\database_scrape\\test_database_csv_to_db.db')
df=pd.read_sql('select * from duplicate_list',conn)
conn.close()

import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_sim(text1, text2):
	docs=[text1,text2]
	tfidf = TfidfVectorizer().fit_transform(docs)
	return ((tfidf * tfidf.T).A)[0,1]
print("cosine")
data_list=[]
for i in range (0,6):
	 text1=df.iloc[i,5]
	 text2=df.iloc[i,6]
	 j=cosine_sim(text1,text2)
	 new_data = {"cosineValues": j}		
	 data_list.append(new_data)
	 with open ('features.csv','w') as file:
        	writer = csv.DictWriter(file, fieldnames = ["cosineValues"])
        	writer.writeheader()
       		for row in data_list:
        		writer.writerow(row)
	 

from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()

def lower_case(title):
     lowercase_tokens = [token.lower() for token in nltk.word_tokenize(title)]
     return lowercase_tokens

def term_overlap(list1, list2):
	s1=lower_case(list1)
	s2=lower_case(list2)
	s3=[]
	s4=[]
	for s in s1:
		s3.append(stemmer.stem(s))
	for s in s2:
		s4.append(stemmer.stem(s))
	intersection = len((set(s3).intersection(s4)))
	union = (len(s3) + len(s4))
	return float((2*intersection) / union)

print("term_overlap")
#data_list=[]
for i in range(0,6):
	list1=df.iloc[i,2]
	list2=df.iloc[i,4]
	k=term_overlap(list1, list2)
	df1 = pd.read_csv("features.csv")
	new_column = df1['cosineValues'] + 1
	df1['TermOverlap'] = new_column
	#df.to_csv('myfile.csv')
	new_data = {"TermOverlap": k}
	data_list.append(new_data)
	with open ('features.csv','w') as file:
        	writer = csv.DictWriter(file, fieldnames = ["cosineValues","TermOverlap"])
        	writer.writeheader()
       		for row in data_list:
        		writer.writerow(row)
	

print("entity overlap")
#jaccard_do
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    print(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)

for i in range(0,6):
	list1=df.iloc[i,2]
	#print(list1)
	list2=df.iloc[i,4]
	#print(list2)
	tokenized_doc1 = nltk.word_tokenize(list1)
	tokenized_doc2 = nltk.word_tokenize(list2)
	# tag sentences and use nltk's Named Entity Chunker
	tagged_sentences1 = nltk.pos_tag(tokenized_doc1)
	tagged_sentences2 = nltk.pos_tag(tokenized_doc2)
	ne_chunked_sents1 = nltk.ne_chunk(tagged_sentences1)
	ne_chunked_sents2 = nltk.ne_chunk(tagged_sentences2)
	# extract all named entities
	named_entities1 = []
	for tagged_tree1 in ne_chunked_sents1:
	    if hasattr(tagged_tree1, 'label'):
	        entity_name1 = ' '.join(c[0] for c in tagged_tree1.leaves()) #
	        entity_type1 = tagged_tree1.label() # get NE category
	        named_entities1.append((entity_name1, entity_type1))
	length1=len(named_entities1)
	print(named_entities1)
	j1=[]
	for i in range(length1):
	       j1.append(named_entities1[i][0])
	print(j1)
	named_entities2 = []
	for tagged_tree2 in ne_chunked_sents2:
	    if hasattr(tagged_tree2, 'label'):
	        entity_name2 = ' '.join(c[0] for c in tagged_tree2.leaves()) #
	        entity_type2 = tagged_tree2.label() # get NE category
	        named_entities2.append((entity_name2, entity_type2))
	length2=len(named_entities2)
	print(named_entities2)
	j2=[]
	for i in range(length2):
	       j2.append(named_entities2[i][0])
	print(j2)
	#jac1=jaccard_similarity(j1, j2)
	#print(jac1)

	#entity type overlap

for i in range(0,6):
	list1=df.iloc[i,2]
	#print(list1)
	list2=df.iloc[i,4]
	#print(list2)
	tokenized_doc1 = nltk.word_tokenize(list1)
	tokenized_doc2 = nltk.word_tokenize(list2)
	# tag sentences and use nltk's Named Entity Chunker
	tagged_sentences1 = nltk.pos_tag(tokenized_doc1)
	tagged_sentences2 = nltk.pos_tag(tokenized_doc2)
	ne_chunked_sents1 = nltk.ne_chunk(tagged_sentences1)
	ne_chunked_sents2 = nltk.ne_chunk(tagged_sentences2)
	# extract all named entities
	named_entities1 = []
	for tagged_tree1 in ne_chunked_sents1:
	    if hasattr(tagged_tree1, 'label'):
	        entity_name1 = ' '.join(c[0] for c in tagged_tree1.leaves()) #
	        entity_type1 = tagged_tree1.label() # get NE category
	        named_entities1.append((entity_name1, entity_type1))
	length1=len(named_entities1)
	print(named_entities1)
	j1=[]
	for i in range(length1):
	       j1.append(named_entities1[0][i])
	print(j1)
	named_entities2 = []
	for tagged_tree2 in ne_chunked_sents2:
	    if hasattr(tagged_tree2, 'label'):
	        entity_name2 = ' '.join(c[0] for c in tagged_tree2.leaves()) #
	        entity_type2 = tagged_tree2.label() # get NE category
	        named_entities2.append((entity_name2, entity_type2))
	length2=len(named_entities2)
	print(named_entities2)
	j2=[]
	for i in range(length2):
	       j2.append(named_entities2[0][i])
	print(j2)
	
	#word_net similarity
	
	from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

def max_wupa(context_sentence, ambiguous_word):
  """ 
  WSD by Maximizing Wu-Palmer Similarity.

  Perform WSD by maximizing the sum of maximum Wu-Palmer score between possible 
  synsets of all words in the context sentence and the possible synsets of the 
  ambiguous words (see http://goo.gl/XMq2BI):
  {argmax}_{synset(a)}(\sum_{i}^{n}{{max}_{synset(i)}(Wu-Palmer(i,a))}

  Wu-Palmer (1994) similarity is based on path length; the similarity between 
  two synsets accounts for the number of nodes along the shortest path between 
  them. (see http://acl.ldc.upenn.edu/P/P94/P94-1019.pdf)
  """

  result = {}
  for i in wn.synsets(ambiguous_word):
    result[i] = sum(max([i.wup_similarity(k) for k in wn.synsets(j)]+[0]) \
                    for j in word_tokenize(context_sentence))
  result = sorted([(v,k) for k,v in result.items()],reverse=True)
  return result

bank_sents = ['I went to the bank to deposit my money',
'The river bank was full of dead fishes']
ans = max_wupa(bank_sents[0], 'bank')
print ans
print ans[0][1].definition
