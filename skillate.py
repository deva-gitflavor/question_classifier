import spacy
import csv
import pandas as pd

nlp_pipe = spacy.load("en_core_web_md")

raw = pd.read_csv('/home/deva/coding_assignment/traininig_dataset (1) (1).txt', sep='/t', header=None)

ab = pd.DataFrame()
for x in range(len(raw)):
	ab.loc[x, 'Question'] = " ".join(raw[0][x].split(':')[1].split(' ')[1:])
	ab.loc[x, 'sub_class'] = raw[0][x].split(':')[1].split(' ')[0]
	ab.loc[x, 'class'] = raw[0][x].split(':')[0]

	post_pipe = nlp_pipe(ab.loc[x, 'Question'])
	words = list(post_pipe.sents)[0]

	ab.loc[x, 'pos'] = words[0].tag_
	ab.loc[x, 'q_word'] = words[0].text
	ab.loc[x, 'two_words'] = str(post_pipe[words[0].i + 0]) + " "+ str(post_pipe[words[0].i + 1])
	ab.loc[x, 'neighbor_pos'] = post_pipe[words[0].i + 1].tag_
	for word in words:
		if word.dep_ == "ROOT":
			ab.loc[x, 'root_pos'] = word.tag_

ab.to_csv('/home/deva/Desktop/midway.csv')
