CS 7641 Project Team 18

# Introduction 
With the advancement of research in the domain of machine learning, one of the active problems is generating sentences with semantic meaning. A lot of progress has been made in the field of NLP to perform tasks such as text classification, language modeling, and natural language understanding. The goal of our project is to generate novel Machine Learning project ideas using transfer learning; and provide recommendations for ML project ideas based on user preferences. 


### Dataset
The dataset that we plan to use consists of titles of all machine learning projects that students at Stanford’s CS229 class submitted over the years 2004 to 2017 (1) and CS230 class from 2018 to 2021(2). It includes ~4200 ideas, each comprising 5-7 words. We further categorize the dataset into ~8 classes that we use for supervised learning classification. Since our dataset is comparatively small, we will use Google's BERT (3), AWD-LSTM (4), and ULMFit (5), which are large language models trained on publicly available corpus of data for transfer learning (6).

# Problem definition 
For courses such as Machine Learning, Deep Learning, and Natural Language Processing, one of the major challenges that students face is not to work on and complete a project, but to come up with a problem statement and a project title. In this project, we plan to solve this problem and generate novel Machine Learning Project ideas. 
Our objective is twofold:
1. To generate a novel machine learning project idea, given a corpus of past ML project ideas.
2. To recommend a machine learning project idea based on a user-input category. 


# Methods
### Objective 1:
Since the dataset that we are working on is relatively small, we aim to use transfer learning to leverage pre-trained large language models such as BERT, AWD-LSTM, and ULMFit. We aim to fine tune these models to learn from our domain specific dataset. This would, in an unsupervised manner, result in the model to generate novel machine learning ideas. 
### Objective 2:
Since our dataset is unlabelled, a preliminary step towards developing supervised learning models would be to cluster the data and generate labels. The approach is to create embeddings for the sentences (ML topics in our case), and use them to cluster the dataset into ~8 categories. We further use this labeled dataset to design a system to recommend an ML project 

