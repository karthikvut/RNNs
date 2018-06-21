import csv
import numpy as np
import itertools
import nltk

#Function to preprocess data
def getSentenceData(path, vocabulary_size=8000):
	unknown_token = "UNKNOWN"
	sentence_start_token = "START"
	sentence_end_token = "END"
	
	#Read the reddits file and append START and END tokens
	print("Reading CSV file")
	with open(path, 'r', encoding='utf-8') as f:
		reader = csv.reader(f, skipinitialspace=True)
		#split full comments into sentences
		sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower() for x in reader])
		#Append START and END tokens
		sentences = ["%s %s %s" % (sentence_start_token, sentence, sentence_end_token) for sentence in sentences]
	print("Parsed %s sentences." % len(sentences))
	
	#Tokenize sentences into words
	tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
	# filter short sentences
	tokenized_sentences = [ x if len(x) > 3 for x in tokenized_sentences ]
	
	#Count word frequencies
	word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
	print("Found %d unique word tokens." % len(word_freq.items))
	
	#Get most common words and build index_to_word and word_to_index vectors
	vocab = word_freq.most_common(vocabulary_size-1)
	index_to_word = [x[0] for x in vocab]
	index_to_word.append(unknown_token)
	word_to_index = dict((w,i) for i,w in enumerate(index_to_word))
	
	print("Vocab size %d."%vocabulary_size)
	print("The least frequent word in our vocabulary is '%s' and appeared %d times." %(vocab[-1][0], vocab[-1][1]))
	
	#Replace words not in vocab with unknown_token
	for i, sentence in enumerate(tokenized_sentences):
		tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sentence]
	
	print("\n Example sentence: '%s'" % sentences[1])
	print("\n Example sentence after preprocessing '%s'" % tokenized_sentences[0])
	
	#Create training data
	X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
	y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
	
	print("X_train shape:" + str(X_train.shape))
	print("y_train shape:" + str(y_train.shape))
	
	#Print training data example
	x_example, y_example = X_train[17], y_train[17]
	print("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
    print("\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example))
	
	return X_train, y_train
	
if __name__ = '__main__':
	X_train, y_train = getSentenceData('data/reddit-comments.csv')
