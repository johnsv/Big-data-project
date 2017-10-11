from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import matplotlib.mlab as mlab
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB



#TRAINING PHASE!!!!!!!!!!!

# Multinomial naive bayes
def naiveBayesClassifier(test, train):

	vectorizer = CountVectorizer(stop_words='english')

	categorizedSentiment = []
	categorizedTest = []
	for val in train[1]:
		categorizedSentiment.append(int(round(val)))

	for val in test[1]:
		categorizedTest.append(int(round(val)))

	# Generate the word matrices for the data
	train_features = vectorizer.fit_transform(train[2])
	test_features = vectorizer.transform(test[2])

	# Fit the naive bayes classifier to the training data using the word counts in the previous lines.
	nb =  MultinomialNB()
	nb.fit(train_features, categorizedSentiment)

	# Generate predictions for the validation data
	predictions = nb.predict(test_features)

	# Compute the ratios of false/true positives
	fpr, tpr, thresholds = metrics.roc_curve(categorizedTest, predictions, pos_label=1)
	
	return f1_score(categorizedTest, predictions, average="micro")



# Multinomial naive bayes using three classes instead of 8
def threeClassesNBT(test, train):
	vectorizer = CountVectorizer(stop_words='english')

	categorizedSentiment = []
	categorizedTest = []
	for val in train[1]:
		if float(val) < -1:
			categorizedSentiment.append(-1)
		elif float(val) > 1:
			categorizedSentiment.append(1)
		else:
			categorizedSentiment.append(0)


	for val in test[1]:
		if float(val) < -1:
			categorizedTest.append(-1)
		elif float(val) > 1:
			categorizedTest.append(1)
		else:
			categorizedTest.append(0)

	# Generate the word matrices for the data
	train_features = vectorizer.fit_transform(train[2])
	test_features = vectorizer.transform(test[2])

	# Fit the naive bayes classifier to the training data using the word counts in the previous lines.
	nb =  MultinomialNB()
	nb.fit(train_features, categorizedSentiment)

	# Generate predictions for the validation data
	predictions = nb.predict(test_features)

	# Compute the ratios of false/true positives
	fpr, tpr, thresholds = metrics.roc_curve(categorizedTest, predictions, pos_label=1)
	
	return f1_score(categorizedTest, predictions, average="micro")



# Gaussian naive bayes
def gausNB(test,train):
	vectorizer = CountVectorizer(stop_words='english')

	categorizedSentiment = []
	categorizedTest = []
	for val in train[1]:
		categorizedSentiment.append(int(round(val)))

	for val in test[1]:
		categorizedTest.append(int(round(val)))

	# Generate the word matrices for the data
	train_features = vectorizer.fit_transform(train[2])
	test_features = vectorizer.transform(test[2])

	denseTrain = train_features.toarray()
	denseTest = test_features.toarray()

	# Fit the naive bayes classifier to the training data using the word counts in the previous lines.
	nb =  GaussianNB()
	nb.fit(denseTrain, categorizedSentiment)

	# Generate predictions for the validation data
	predictions = nb.predict(denseTest)

	# Compute the ratios of false/true positives
	fpr, tpr, thresholds = metrics.roc_curve(categorizedTest, predictions, pos_label=1)

	i = 0
	error = 0
	for x in predictions:
		error = error + abs(x - categorizedTest[i])
		i = i+1

	return (error/419)

	
	#return f1_score(categorizedTest, predictions, average="micro")



#SVM gridsearch to find
def svc(test,train):
	vectorizer = CountVectorizer(stop_words='english')
	featuresTrain = train[2]
	featuresTest = test[2]
	
	catLabelTrain = []
	catLabelTest = []
	for val in train[1]:
		catLabelTrain.append(int(round(val)))

	for val in test[1]:
		catLabelTest.append(int(round(val)))
	


	#train_features = vectorizer.fit_transform(train[2])
	#param_grid = [
	  #{'C': [0.01,0.05,0.1,0.2,0.3], 'kernel': ['linear']},
	#  {'C': [10], 'gamma': [0.001, 0.005,0.01,0.02,0.03,], 'kernel': ['rbf']},
	# ]

	# Perform the gridsearch with 10-fold-cross-validation to select the best parameters for our dataset.
	#model = svm.SVC()
	#gridsearch = GridSearchCV(model, param_grid)
	#gridsearch.fit(train_features, categorizedSentiment)
	#print(gridsearch.best_params_) #Returns best parameters

	train_features = vectorizer.fit_transform(featuresTrain)
	test_features = vectorizer.transform(featuresTest)

	clf = SVC(kernel="rbf", C=10, gamma=0.01) #training using best parameters from gridsearch
	clf.fit(train_features, catLabelTrain)

	predictions = clf.predict(test_features)

	i = 0
	error = 0
	for x in predictions:
		error = error + abs(x - catLabelTest[i])
		i = i+1

	return (error/419)

	#return f1_score(catLabelTest, predictions, average="micro")







def nn(test,train):
	vectorizer = CountVectorizer(stop_words='english')
	featuresTrain = train[2]
	featuresTest = test[2]
	
	catLabelTrain = []
	catLabelTest = []
	for val in train[1]:
		catLabelTrain.append(int(round(val)))

	for val in test[1]:
		catLabelTest.append(int(round(val)))
	


	train_features = vectorizer.fit_transform(featuresTrain)
	test_features = vectorizer.transform(featuresTest)

	#print(train_features)

	clf = MLPClassifier(solver='lbfgs',  
                    hidden_layer_sizes=(50,50))

	clf.fit(train_features, catLabelTrain)

	predictions = clf.predict(test_features)

	i = 0
	error = 0
	for x in predictions:
		error = error + abs(x - catLabelTest[i])
		i = i+1

	return (error/419)

	#return f1_score(catLabelTest, predictions, average="micro")


def knn(test,train):

	vectorizer = CountVectorizer(stop_words='english')
	featuresTrain = train[2]
	featuresTest = test[2]
	
	catLabelTrain = []
	catLabelTest = []
	for val in train[1]:
		catLabelTrain.append(int(round(val)))

	for val in test[1]:
		catLabelTest.append(int(round(val)))
	


	train_features = vectorizer.fit_transform(featuresTrain)
	test_features = vectorizer.transform(featuresTest)

	clf = KNeighborsClassifier(n_neighbors=3)

	clf.fit(train_features, catLabelTrain)

	predictions = clf.predict(test_features)


	return f1_score(catLabelTest, predictions, average="micro")


	# Print prediction along with the review
	#i = 0
	#numOfError = 0

#	error = 0
#	for x in predictions:
#		error = error + abs(x - categorizedTest[i])
#		i = i+1

#	return (error/419)


train = pd.read_csv('tweets_GroundTruth.txt', sep="\t", header=None)
test = pd.read_csv('testBookings.csv', sep=";", header=None)
dataSize = len(train)


#################### CV ##########################################


cvSplit = []
foldSize = int(dataSize/10)-1

for i in range(0, 10):
	new = train[(i*foldSize):(i*foldSize+foldSize):1]
	cvSplit.append(new)

error = 0
svmMeanF1 = 0
nbMeanF1 = 0
mlpMeanF1 = 0
knnMeanF1 = 0
gausMeanF1 = 0
threeClassesNB = 0
#cross-validation
for i, validationSet in enumerate(cvSplit):
	trainData = []
	frames = pd.DataFrame()
	list_ = []

	for j, newData in enumerate(cvSplit):
		if j != i:
			list_.append(newData)
	frames = pd.concat(list_)
	nbMeanF1 += naiveBayesClassifier(validationSet, frames)
	gausMeanF1 += gausNB(validationSet, frames)
	threeClassesNB += threeClassesNBT(validationSet, frames)
	svmMeanF1 += svc(validationSet, frames)
	mlpMeanF1 += nn(validationSet, frames)
	knnMeanF1 += knn(validationSet, frames)
	
	print("Tested:  " + repr(i))

	i = i+1



	
###############################################################


print("NB F1 score " + repr(nbMeanF1/10));
print("SVM F1 score " + repr(svmMeanF1/10));
print("MLP F1 score " + repr(mlpMeanF1/10));
print("KNN F1 score " + repr(knnMeanF1/10));
print("Gaussian Nb F1 score " + repr(gausMeanF1/10));
print("three classes Nb F1 score " + repr(threeClassesNB/10));


# APPLICATION PHASE!!!!!!!!!!!!!!!!!!
#Best model SVM using kernel = rbf, C = 10 and gamma = 0.01



#Uses the best model to predict each day
def predictSentiment(testsetName, bestModel, totalMean):
	testSubset = pd.read_csv(testsetName, sep="\n", header=None)
	print(testsetName + "\n(" + repr(len(testSubset)) + " items)")

	predictions = bestModel.predict(vectorizer.transform(testSubset[0]))
	predictionSum = 0
	n = 0
	squaredDiff = 0
	pos = 0
	neg = 0
	neu = 0
	for pred in predictions:
		predictionSum = predictionSum + pred
		squaredDiff = squaredDiff + (pred - totalMean)**2
		if pred > 0:
			pos = pos+1
		elif pred == 0:
			neu = neu+1
		else:
			neg = neg+1


	print("Average prediction score (" + testsetName + "): " + repr(predictionSum/len(test)) + "\n" + "squaredDiff: " + repr(squaredDiff))
	print("Positive: " + repr(pos) + "\nNegative: " + repr(neg) + "\nNeutral: " + repr(neu))
	return (predictionSum/len(test))




## MAIN
categorizedSentiment = []
for val in train[1]:
	categorizedSentiment.append(int(round(val)))
train_features = vectorizer.fit_transform(train[2])
model =  svm.SVC()
model = svm.SVC(kernel=bestParams.get('kernel'), C=bestParams.get('C'), gamma=bestParams.get('gamma'))
model.fit(train_features, categorizedSentiment)

totalMean = 0.3401630411
predictSentiment("MonTwitterSentiment.txt", model, totalMean)
predictSentiment("TueTwitterSentiment.txt", model, totalMean)
predictSentiment("WedTwitterSentiment.txt", model, totalMean)
predictSentiment("ThuTwitterSentiment.txt", model, totalMean)
predictSentiment("FriTwitterSentiment.txt", model, totalMean)
predictSentiment("SatTwitterSentiment.txt", model, totalMean)
predictSentiment("SunTwitterSentiment.txt", model, totalMean)




# the histogram of the data
#n, bins, patches = plt.hist(sorted(train[1]), 50, normed=1, facecolor='green', alpha=0.75)
#mu, sigma = 100, 15
# add a 'best fit' line
#y = mlab.normpdf( bins, mu, sigma)
#l = plt.plot(bins, y, 'r--', linewidth=1)
#plt.show()
#plt.hist(sorted(train[1]), 50)
#plt.show()













