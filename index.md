CS 7641 Machine Learning Fall 22
Project Team 18

# Contents
1. [Introduction](#introduction)
    1. [Dataset](#dataset)
2. [Problem Definition](#problem-definition)
3. [Methods](#methods)
    1. [Objective 1](#objective-1-unsupervised)
    2. [Objective 2](#objective-2-supervised)
4. [Results](#results)
    1. [Unsupervised](#unsupervised)
    2. [Supervised](#supervised)
5. [References](#references)
6. [Gantt Chart](#gantt-chart)
7. [Contributions](#contribution-table)
8. [Presentation Video](#presentation-video)

# Introduction 
With the advancement of research in the domain of machine learning, one of the active problems is generating sentences with semantic meaning. A lot of progress has been made in the field of NLP to perform tasks such as text classification, language modeling, and natural language understanding. BERT[1] is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both the left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks. ULMFiT[2] proposed an effective transfer learning method that can be applied to any task in NLP, and introduce techniques that are key for fine-tuning a language model. The paper which introduces AWD-LSTM[3], also proposed ways to investigate strategies for regularizing and optimizing LSTM-based models for the specific problem of word-level language modeling - which means building a model which can calculate the joint level probability of what the next word should be, given a sequence of words. The approach and results displayed using AWD-LSTM were our main source of inspiration behind our project. In this project, we plan to leverage and build on top of similar NLP research and fine-tune pre-trained models to generate novel Machine Learning project ideas using transfer learning and classify ML project ideas into different labels. 



### Dataset
The dataset that we plan to use consists of titles of machine learning projects that students at Stanford’s CS229 class submitted over the years 2004 to 2016 [4] and CS230 class from 2018 to 2021 [5]. It includes 4388 ideas, and we further categorize the dataset into 8 classes that we use as ground truth labels for supervised classification.

# Problem definition 
For courses such as Machine Learning, Deep Learning, and Natural Language Processing, one of the major challenges that students face is to come up with a problem statement or project title. We were also in a similar situation and that's when we thought of solving this problem. 
Our objective is twofold:
1. Unsupervised: To generate a novel machine learning project idea, given a corpus of past ML project ideas**(done)**
2. Supervised: To classify a machine learning project idea into different human-labeled categories such as NLP, Vision, Robotics, Health, Finance, Entertainment, Game AI and Generic ML**(next step)**

# EDA

# Data Collection
For data collection, we decided to use the publicly available datasets consisting of machine learning project ideas and the best one that we found was the collection of titles of projects done by students at Stanford class CS229 and CS230. We created our dataset after extracting only the titles from websites and saved it as an excel file. We did not use CSV file format, since the titles themselves contain ‘commas’ and that corrupts the dataset. The entire corpus of data contains 4388 project ideas/titles. We used 70% of the data for training and 30% as validation data for the unsupervised model training. Once all the titles were extracted, we manually classified the titles into predefined categories such as NLP, Vision, Robotics, Health, Finance, Game AI, Entertainment, and Generic ML, which will be used for training a supervised model. We did this by going through the labels manually or reading the abstracts if the titles were not reflective of the content. 

### Data Preprocessing
After we created the dataset, the next step was to preprocess and clean the data. We process the data to extract only useful information and remove any inconsistencies such as missing values, noisy data, or non-informative features. We remove all the duplicate project titles, by first converting them into lowercase and then scraping off the duplicate values. Given that our dataset consists of project titles, we also eliminated all project ideas with less than 3 words as input. The idea behind this is that project titles with just one or two words, for example, “Bootstrapping”, “Al Mahjong”, “MindMouse”, “rClassifier” etc.,  were not very descriptive and did not seem to add much value to the training dataset. They might even further corrupt the dataset in the case of supervised learning. We further processed the dataset and removed the stopwords such as “and”, “of”, “the” etc. However, in our case, removing these resulted in the project title ideas losing their meanings which were required for our processing since it might have resulted in. For example, input such as “A System for Segmenting Video of Juggling” would become “System Segmenting Video Juggling” and might result in a Garbage in Garbage out situation. Thus, we did not use this as a feature for our dataset. 


# Methods
### Objective 1 (Unsupervised):
Since the dataset that we are working on is relatively small, we have used transfer learning [6] to leverage the pre-trained large language model AWD-LSTM. This model is trained on publicly available textual corpus (such as parliamentary records or Wikipedia) and implicitly encodes knowledge of the English language. It is a type of recurrent neural network that has been optimized and trained using techniques like DropConnect for regularization, NT-ASGD for optimizations and many more. We then create a language model fine-tuned for our dataset with the pre-trained weights of AWD-LSTM. We first trained the last layers and left most of the model exactly as it was. To improve our model further, we didn’t unfreeze the whole data but unfreeze one layer at a time starting with the last 2 layers, then the next layer, and finally the entire model for 20 epochs. We then generated 10 ideas and evaluated the results according to our metrics.

### Objective 2 (Supervised):
Our dataset is unlabelled, thus, a preliminary step toward developing supervised learning models would be to tag those titles/ideas manually. We further feed this labeled data to machine learning algorithms such as SVM, Multi-class regression, XGBoost, Naive Bayes Classifier, and random forest to train them. During the testing phase, we input a machine learning project idea to the algorithm, and it classifies it into human-labeled categories.



# Results and Discussion

For POC we first generate only 10 ideas from the model that we trained on the dataset (can be easily extended, since it depends on the user input on how many generated novel ideas the user wants)
![results](results.jpg)<div align ="center">*Results*</div>

# References

1. [Stanford Projects](https://cs229.stanford.edu/projects2016)
2. [Past Projects](http://cs230.stanford.edu/past-projects/)
3. [Transfer Learning for Style-Specific Text Generation](https://nips2018creativity.github.io/doc/Transfer%20Learning%20for%20Style-Specific%20Text%20Generation.pdf)
4. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
5. [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182v1)
6. [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
7. [Glue: A MultiTask Benchmark and Analysis Platform for Natural Language Understanding](https://openreview.net/pdf?id=rJ4km2R5t7)
8. [Evaluation Metrics for Multi-Label Classification](https://medium.datadriveninvestor.com/a-survey-of-evaluation-metrics-for-multilabel-classification-bb16e8cd41cd)



# [Gantt Chart](https://docs.google.com/spreadsheets/d/1Ckuu6r8BdbIab1lo3kJkdjhAnlVZ6WLj/edit#gid=422388448)

# Contribution Table

![Table](table.JPG)

# [Presentation Video](https://www.youtube.com/watch?v=fOmfPSxn8Qg)
