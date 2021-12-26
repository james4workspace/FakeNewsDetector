# FakeNewsDetector
Project from Kaggle :https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

## 1. Step 1: Preprocessing Part

The DataSet was separated into `True.csv` and `Fake.csv` already. After importing the data, we can see the data structure is like below:

| index | title  | text | subject | date |
| ------------- | ------------- | ------------- | ------------- |------------- |
| type: integer | type: string | type: string | type: string | type: string|

(1) It's obvious that the `subject` containing limited choices, which should be taken as categorical attribute. Therefore, in preprocessing part, I transfered `subject` by using one hot encoding.

(2) And I also transfered date into different attributes: year, month, day and the judge of April Fool's Day. Because people are prone to produce fake news on April Fool's Day.

(3) Next step is to clean the title and text, in order to achieve high coverage of text that can be transfered into vectors based on GloVe pre-trained DataSet.

(4) Finally, for the purpose of making deep learning model easier to determine the different part of dataframe, I merged the cleaned data together by the order of: date related attributes, subject related attributes, title, text and label.

## 2. Step 2: GloVe Word-Embedding

For most machine learning algorithms such as SVM, XGBoost, each attribute must contain some significance for comparison. Thus purely changing the word into vectors with different length of word list is not a good idea for further training. So I transfered the title and text into vectors, and only store the mean of their vectors as new attributes, and merge them with date related attributes and subject related attributes.

For Deep Leaning Model such as LSTM, CNN, it's better to store all the vectors with different length of word list. But for better learning, I set one new attribute named as `mark for title` with same content `end of title` to separate the title part and text part easily, and purly transfered date related attributes and subject related attributes into vectors.

## 3. Step 3: XGBoost, 1dCNN, BiLSTM

For training models part, I chose XGBoost as baseline model, because normally XGBoost works pretty well in such topics by categorizing attributes as multiple trees for classification. And I chose BiLSTM based on my previous experience, because time-series deep learning model can always work well in NLP projects. And if we take text as the sentence spoken by someone, it's obvious that the context changed by the time. Therefore, considering the former part of text can be very important for training. Choosing 1dCNN is also according to the same reason, because if we take text as a whole picture, we can different parts of the sentence make up the final significance of the sentence. Thus, CNN, especially 1dCNN can work pretty well in NLP project.



