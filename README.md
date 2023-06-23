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

# Data Preprocessing and Vectorization
The text data was vectorized using the TF-IDF (Term Frequency-Inverse Document Frequency) approach. The `TfidfVectorizer` was employed to convert the text data into numerical feature vectors. TF-IDF assigns weights to each term in the corpus based on its frequency in a document and its rarity in the entire corpus. This  allows the models to work with numerical representations of text data, capturing the importance of words in the documents.

Initially, no preprocessing of the data was performed while training the first couple of models.
Lateron the Twitter text data was preprocessed to clean and transform the text. The preprocessing included the following steps:

## N-grams
In order to include some context to each word, I tokenized using n-gram ranges multiple n-gram ranges. The best model was an SVM classifier using an n-gram range of (1,7) without any other preprocessing with an F1-Score of 0.57.

## Stopword removal
The removal of noise did not result in substantial improvement of the model. With an F1-Score of 0.55 it was a worse strategy than n-grams. This is also understandable, since stopwords seem detremental when analyzing sentiment of a tweet.

## Feature Union
Using Feature Untion and Pipeline I create a pipeline using feature union, which combines two different TfidfVectorizer transformers with different n-gram ranges and lowercase settings. This experiment did not result in any substatial improvement of the model. This model has an F1-Score of 0.54.

## Processing Tweet
Even though the tweets are in english, the language of the user is not standard. To standardize the syntax of the tweets to an extent, I created a function to process the tweets and only extract relevant text. The following steps were made:

- removing URLs, usernames, hashtags, punctuation
- tokenizing the text using TweetTokenizer
- lemmatizing the tokens

These steps aimed to remove noise, standardize the text, and improve the quality of the input data. In reality they did not result in a major improvement of the model. The F1-score with or without lemmatization stayed at 0.56.

# Model Training and Evaluation
I trained and evaluated both the Naive Bayes and SVM classifiers using the vectorized data. The classifiers were implemented using scikit-learn library. 
To counter the imbalance of the classes I used the F1-Score as the evaluation metric. 

For the Naive Bayes classifier, I used the MultinomialNB class with an alpha value of 0.1. The classifier was trained on the vectorized training data and evaluated on the test data. 
With an F1-Score of 0.53 this model was the worst performing model.

For the SVM classifier, I utilized the LinearSVC class, which is a linear support vector classifier. The classifier was trained on the vectorized training data and evaluated on the test data using the same metrics as the Naive Bayes classifier. For this classifier I experimented with many different methods to improve the model performance.

# Error Analysis
To perform a thorough error analysis, I examined the predictions made by both classifiers on individual test samples. I analyzed cases where the classifiers' predictions differed and explored the reasons behind the misclassifications. I also compared the predictions of each model to identify any patterns or trends that could provide insights into potential improvements.

# Fine-Tuned Transformer
For the fine-tuned model I created a custom dataset TweetDataset for the tweets and labels. This TweetDataset also includes the processing function defined in the previous chapter (without lemmatizing). The reason being that a bert based language model works better without the noisy characters in a tweet.

As learning rate for the optimizer I used 2e-5, which is small, but for fine tuning a model I deemed it appropriate, since they are sensitive to large changes in their weights.

As batch size for my dataset I used 15. 

I trained the model for 5 epochs. 

For each epoch I calculated the following metrics:
- Training loss: For each batch I calculated the total loss as sum of the loss for each iteration and devided it by the number of batches for the average loss per epoch
- Validation Loss: Same as Training Loss but for the validation loop
- Validation Accuracy: correctly predicted labels of the validation data / total validation data
- Validation F1-Score: F1 Score of the validation data

After the second Epoch the Training loss was still declining while the Validation Loss started to increase, which means the model started to overfit the training dataset. Yet the Validation F1-Score still increased in the 3. epoch with f1=0.7146.

# Proposed Improvements
## Text Preprocessing
Further advanced Text preprocessing could be done to handle specific cases.

## Feature Engineering
Even more information can be extracted from the tweets. For example word embeddings, part-of-speech tags, named entity recognition or sentiment scores could provide the model with more information.

## Hyperparameter tuning
Using techniques like GridSearch ideal parameters could be identified to improve the model performance.

# Conclusion


# Reflexion
