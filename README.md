# nlp-2023-mc1


# Introduction
The goal of this project was to evaluate and compare two different text classification systems (https://spaces.technik.fhnw.ch/spaces/natural-language-processing/aufgaben). The task involved building, training, and evaluating the systems, performing a thorough error analysis, and proposing theories for system improvements.

# Dataset
The dataset used for this project was a Twitter text dataset with sentiment labels (https://huggingface.co/datasets/tweet_eval). The dataset consisted of three subsets: training, testing, and validation. The training set had 45,615 rows, the test set had 12,284 rows, and the validation set had 2,000 rows. The sentiment labels in the dataset were categorized as 0 (negative), 1 (neutral), and 2 (positive).

# Model Selection
For this text classification task, I chose to evaluate two different classification systems: Naive Bayes and Support Vector Machines (SVM).

The Naive Bayes classifier is based on the probabilistic principle of Bayes' theorem. It assumes that the features are conditionally independent given the class labels. Naive Bayes is known for its simplicity, efficiency, and effectiveness, but it is also limited regarding complexity of the problem.

SVM, on the other hand, works by mapping the input data into a high-dimensional feature space and finding the best decision boundary. SVMs are particularly useful when dealing with complex classification problems.

Lastly a distilbert-base-uncased pretrained model from HuggingFace (https://huggingface.co/distilbert-base-uncased) was finetuned with our training data. 

# Data Preprocessing
Initially, no preprocessing of the data was performed while training the first couple of models.

Lateron the Twitter text data was preprocessed to clean and transform the text. The preprocessing steps included removing URLs, usernames, and hashtags, tokenizing the text, removing stopwords and punctuation, and lemmatizing the tokens. These steps aimed to remove noise, standardize the text, and improve the quality of the input data.

# Vectorization
The text data was vectorized using the TF-IDF (Term Frequency-Inverse Document Frequency) approach. The `TfidfVectorizer` was employed to convert the text data into numerical feature vectors. TF-IDF assigns weights to each term in the corpus based on its frequency in a document and its rarity in the entire corpus. This  allows the models to work with numerical representations of text data, capturing the importance of words in the documents.

# Model Training and Evaluation
I trained and evaluated both the Naive Bayes and SVM classifiers using the vectorized data. The classifiers were implemented using scikit-learn library. 
To counter the imbalance of the classes I used the F1-Score as the evaluation metric. 

For the Naive Bayes classifier, I used the MultinomialNB class with an alpha value of 0.1. The classifier was trained on the vectorized training data and evaluated on the test data. 
With an F1-Score of 0.53 this model was the worst performing model.

For the SVM classifier, I utilized the LinearSVC class, which is a linear support vector classifier. The classifier was trained on the vectorized training data and evaluated on the test data using the same metrics as the Naive Bayes classifier. For this classifier I experimented with many different methods to improve the model performance.

# Error Analysis
To perform a thorough error analysis, I examined the predictions made by both classifiers on individual test samples. I analyzed cases where the classifiers' predictions differed and explored the reasons behind the misclassifications. I also compared the predictions of each model to identify any patterns or trends that could provide insights into potential improvements.

# Proposed Improvements


# Conclusion


# Reflexion
