#!/usr/bin/env python
# coding: utf-8

# ## Supervised Learning
# ## Project: Finding Donors for *CharityML*

# ----
# ## Exploring the Data
# 

# In[1]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the Census dataset
data = df = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=1))


# ### Implementation: Data Exploration
# 
#  

# Let us take a broad view at our dataset

# In[2]:


df


# Let's take a closer look at the available columns

# In[3]:


df.describe()


# Looking into the income column

# In[4]:


df.income.describe()


# In[5]:


df.income.value_counts()


# Now, let us calculate the percentage of individuals whose income is more than $50,000

# In[6]:


11208/45222*100


# In[7]:


# TODO: Total number of records
n_records = 45222

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = 11208

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = 34014

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = 24.78

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))


# ----
# ## Preparing the Data
# 

# ### Transforming Skewed Continuous Features
# 

# In[8]:


# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)


# In[9]:


# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)


# ### Normalizing Numerical Features
# 

# In[10]:


# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))


# In[11]:


df2=features_log_minmax_transform
df2


# ### Implementation: Data Preprocessing
# 

# In[12]:


# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

# TODO: Encode the 'income_raw' data to numerical values
income = income_raw.apply(lambda x: 1 if x == ">50K" else 0)

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
print(encoded)


# ### Shuffle and Split Data
# 

# In[13]:


# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# ----
# ## Evaluating Model Performance
# 

# ### Metrics and the Naive Predictor
# 
# 

# ### Question 1 - Naive Predictor Performace
# * If we chose a model that always predicted an individual made more than $50,000, what would  that model's accuracy and F-score be on this dataset? 

# In[14]:


'''
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''
TP = np.sum(income)
FP = income.count() - TP
FN = 0

# TODO: Calculate accuracy, precision and recall
A = accuracy = TP/(TP+FP)
R = recall = TP/(TP+FN)
P = precision = TP/(TP+FP)

# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
B = 0.5
fscore = (1+(B*B)) * ((P * R) / ((B*B*P)+R))

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


# Therefore, if we chose a model that always predicted an individual made more than $50,000, our model's accuracy and F-score would be 0.2478 and 0.2917 respectively.

# ###  Supervised Learning Models
# **The following are some of the supervised learning models that are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
# - Gaussian Naive Bayes (GaussianNB)
# - Decision Trees
# - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K-Nearest Neighbors (KNeighbors)
# - Stochastic Gradient Descent Classifier (SGDC)
# - Support Vector Machines (SVM)
# - Logistic Regression

# # Question 2 - Model Application
# List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen
# 
# - Describe one real-world application in industry where the model can be applied. 
# - What are the strengths of the model; when does it perform well?
# - What are the weaknesses of the model; when does it perform poorly?
# - What makes this model a good candidate for the problem, given what you know about the data?
# 
# 

# **Answer: **
# ## 1. Decision Tree
# 
# - Application: Decision tree model can be used in schools to determine whether or not a student will be admitted based on say                  grades and test score
# - Strenght: A decision tree does not require normalization of data.
# - Weakness: The decision tree model often overfits
# - The model is good for use here because we are dealing with a classification problem
# 
# ## 2. Ensemble Methods:AdaBoost
# 
# - Application: It is commoly used for wight voting and averaging
# - Strength: Ensemble methods help in to reduce bias and variance
# - Weakness: It might be difficult to master, and any incorrect choice might result in a model with lower prediction accuracy               than an individual model.
# - The model is good for use here because it will prevent overfitting 
# 
# ## 3. Support Vector Machines(SVM)
# 
# - Application: It can be used for medical diagnosis and object dectection
# - Strength: SVM works relatively well when there is a clear margin of separation between classes.
# - Weakness: SVM does not perform very well when the data set has more noise i.e. target classes are overlapping.
# - This model is good for use here because SVM can be used for both regression and classification problem
# 
# 
# 

# ### Implementation - Creating a Training and Predicting Pipeline
# To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data.

# In[15]:


# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end- start
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
        
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results


# ### Implementation: Initial Model Evaluation
# 

# In[16]:


# TODO: Import the three supervised learning models from sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# TODO: Initialize the three models
clf_A = AdaBoostClassifier(random_state=1)
clf_B = SVC(random_state=1)
clf_C = DecisionTreeClassifier(random_state=1)

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
samples_100 = len(y_train)
samples_10 = int(len(y_train)/10)
samples_1 = int(len(y_train)/100)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)


# ----
# ## Improving Results
# In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F-score. 

# ### Question 3 - Choosing the Best Model
# 
# * Based on the evaluation you performed earlier, in one to two paragraphs, explain to *CharityML* which of the three models you believe to be most appropriate for the task of identifying individuals that make more than \$50,000. 
# 

# **Answer: **
# 
# As observed on the grapgh above; considering prediction and training time, Adaboost and DecisionTree are the better classifiers. For F-score, when we the testing set was 100%, Adaboost has the highest F-score, followed by SVC then DecisionTree. 
# Hence, Adaboost is the most suitable algorithm for our data since we are interested in the classifier with the hghest F-score.

# ### Question 4 - Describing the Model in Layman's Terms
# 
# * In one to two paragraphs, explain to *CharityML*, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical jargon, such as describing equations.
# 

# **Answer: ** 
# 
# The Adaboost algorithm was discovered by Freund and Schapire in 1996. In Adaboost, we combine a number of week learner to form a strong with greater accuracy than the constituiting weal learners. AdaBoost is a quick learning method that works with big data sets and may be used with a number of learning technique. From a given data, the algorithm tries to classify and predict various features in the dataset. After the classification, the algorithm checks for the misclasified features and the reclassify the data giving priority to the the misclassified points. This process continues on and on, the weak learning models are combine to obtain a strong learning model, which will better fit our data.
# 

# ### Implementation: Model Tuning
# Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this.

# In[17]:


# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostClassifier

# TODO: Initialize the classifier
clf = AdaBoostClassifier(random_state=40)

# TODO: Create the parameters list you wish to tune, using a dictionary if needed.
parameters = {'n_estimators' : [50,75,100,200], 'learning_rate' : [0.1,0.5,1.0]}

# TODO: Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train,y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))


# ### Question 5 - Final Model Evaluation
# 
# * What is your optimized model's accuracy and F-score on the testing data? 
# * Are these scores better or worse than the unoptimized model? 
# * How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in **Question 1**?_  
# 
# 

# #### Results:
# 
# |     Metric     | Unoptimized Model | Optimized Model |
# | :------------: | :---------------: | :-------------: | 
# | Accuracy Score |       0.8576      |     0.8651      |
# | F-score        |       0.7246      |     0.7396      |
# 

# **Answer: **
# 
# Unoptimized model
# ------
# Accuracy score on testing data: 0.8576
# F-score on testing data: 0.7246
# 
# Optimized Model
# ------
# Final accuracy score on the testing data: 0.8651
# Final F-score on the testing data: 0.7396
# 
# The score is better than the unoptimized model
# 
# For the Naive Predictor, Accuracy score = 0.2478 and F-score = 0.2917. The optimized model has a far greater accuracy and F-score compared to that of the Naive Predictor. Hence, the optimized model is better than the Naive Predictor

# ----
# ## Feature Importance
# 
# 

# ### Question 6 - Feature Relevance Observation
# When **Exploring the Data**, it was shown there are thirteen available features for each individual on record in the census data. Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?

# **Answer:**
# 
# The 5 features that I believe should be most importanr for prediction are age, occupation, capital gain, capital loss and hours-per-week, in that order
# 
# 1. Age: Age in my opinion is the most important feature upon which all other features depend
# 
# 2. Occupation: The level of income of an individual is dependent on the individual's occupation, whether he/she works at a private firm, local goverment or working without pay
# 
# 3. Capital gain: The amount of mometary capital gain of an individual is also very important feature. The higher the capital gain, the higher the income
# 
# 4. Capital loss: The more the monetary capital loss of an individual, the lower the individual's income
# 
# 5. Hours-per-week: The average number of hours worked by an individual is also critical to the individual's income. All other things held constant, the more hours an individual works the higher his/her income will be.

# ### Implementation - Extracting Feature Importance
# Choose a `scikit-learn` supervised learning algorithm that has a `feature_importance_` attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.
# 

# In[18]:


# TODO: Import a supervised learning model that has 'feature_importances_'
from sklearn.ensemble import AdaBoostClassifier

# TODO: Train the supervised model on the training set using .fit(X_train, y_train)
model = AdaBoostClassifier(random_state = 1).fit(X_train, y_train)

# TODO: Extract the feature importances using .feature_importances_ 
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)


# ### Question 7 - Extracting Feature Importance
# 
# Observe the visualization created above which displays the five most relevant features for predicting if an individual makes at most or above \$50,000.  
# * How do these five features compare to the five features you discussed in **Question 6**?
# * If you were close to the same answer, how does this visualization confirm your thoughts? 
# * If you were not close, why do you think these features are more relevant?

# **Answer:**
# 
# - Four of the five features predicted in Question 6 were right. Just that in the plot above, education-num replaced occupation.
# 
# - The prediction was close, just that the number of educational years completed surprisingly is an impoertant feature affecting income. 

# ### Feature Selection
# How does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of **all** features present in the data. This hints that we can attempt to *reduce the feature space* and simplify the information required for the model to learn. The code cell below will use the same optimized model you found earlier, and train it on the same training set *with only the top five important features*. 

# In[19]:


# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))


# ### Question 8 - Effects of Feature Selection
# 
# * How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?
# * If training time was a factor, would you consider using the reduced data as your training set?

# **Answer:**
# 
# -  Unfortunately, the final model's F-score and accuracy score on the reduced data is lower compared to that of the full data
# 
# - If training time is of utmost importance, I would rather consider using the reduced data as my training set, since the difference in accuracy and F-score between the full and reduced model is not that profound.
