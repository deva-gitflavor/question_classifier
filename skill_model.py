from __future__ import unicode_literals
import numpy as np
from sklearn.svm import LinearSVC
import pandas as pd
import spacy
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import GaussianNB

nlp_pipe = spacy.load("en_core_web_sm")

train = pd.read_csv('./midway.csv')
y = train.pop('class')
train.pop('sub_class')
train.pop('Question')
train.pop('two_words')

X_train = pd.get_dummies(train)

# training_file = pd.read_csv('/home/deva/coding_assignment/traininig_dataset (1) (1).txt', sep='/t', header=None)
raw = pd.read_csv('./validation_dataset (1) (1).txt', encoding='utf-8', sep='/t', header=None)


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



# get_spacy_tokens(training_file)

compare = ab.pop('class')
ab.pop('sub_class')
ab.pop('Question')
ab.pop('two_words')

X_predict = pd.get_dummies(ab)


def transform_data_matrix(X_train, X_predict):
    X_train_columns = list(X_train.columns)
    X_predict_columns = list(X_predict.columns)

    X_trans_columns = list(set(X_train_columns + X_predict_columns))
    # print(X_trans_columns, len(X_trans_columns))

    trans_data_train = {}

    for col in X_trans_columns:
        if col not in X_train:
            trans_data_train[col] = [0 for i in range(len(X_train.index))]
        else:
            trans_data_train[col] = list(X_train[col])

    XT_train = pd.DataFrame(trans_data_train)
    XT_train = csr_matrix(XT_train)
    trans_data_predict = {}

    for col in X_trans_columns:
        if col not in X_predict:
            trans_data_predict[col] = 0
        else:
            trans_data_predict[col] = list(X_predict[col])  # KeyError

    XT_predict = pd.DataFrame(trans_data_predict)
    XT_predict = csr_matrix(XT_predict)

    return XT_train, XT_predict

X_train, X_predict = transform_data_matrix(X_train, X_predict)

#def support_vector_machine(X_train, y, X_predict):
#     lin_clf = LinearSVC()
#     lin_clf.fit(X_train, y)
#     prediction = lin_clf.predict(X_predict)
#     return prediction

def naive_bayes_classifier(X_train, y, X_predict):
    gnb = GaussianNB()
    X_predict = X_predict.todense()
    X_train = X_train.todense()
    gnb.fit(X_train, y)   
    prediction = gnb.predict(X_predict)
    return prediction


#predicted = support_vector_machine(X_train, y, X_predict)
predicted = naive_bayes_classifier(X_train, y, X_predict)


final_df = pd.DataFrame(predicted, columns=['predicted'])
final_df['actual'] = compare

for i in range(len(final_df)):
    if final_df.loc[i, 'predicted'] == final_df.loc[i, 'actual']:
        final_df.loc[i, 'validate'] = 1
    else:
        final_df.loc[i, 'validate'] = 0
        

accuracy = (sum(final_df['validate'])/len(final_df))*100
print(final_df)
print("Accuracy:",accuracy)
