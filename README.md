# Twitter-Sentiment-Analysis-on-Mask-Mandate-Policy-in-NYC
###### __Author__: Abdullah Altammami; Jing Gao; Nathanael George
###### __*Weill Cornell Graduate School of Medical Sciences, New York, United States*__


> Table of content:
- [Introduction](#introduction)
- [Method](#method)
  * [Dataset introduction](#dataset-introduction)
  * [Data annotation](#data-annotation)
  * [Browsing the dataset](#browsing-the-dataset)
  * [Preprocess the data](#preprocess-the-data)
  * [Data Splitting](#data-splitting)
  * [Fit the classifiers with preprocessed data](#fit-the-classifiers-with-preprocessed-data)
- [Result](#result)
  * [BernoulliNB](#bernoullinb)
  * [RandomForest](#randomforest)
  * [SVC](#svc)
  * [KNN](#knn)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [Reference](#reference)




## Introduction
---

* Tracking public health trends and identifying peaks of public concern are crucial tasks for researchers. Moreover, monitoring such concerns may not be cost-effective and expensive to perform with traditional surveillance systems and have limited coverage and expected delays. Using Twitter user data to understand public health trends is a method used to gain insights on the public’s perception of health crises.
* Sentiment analysis is a natural language processing application and the goal of its use is to provide an automatic classification of the sentiments in the unstructured text data. In our study, the objective was to extract Twitter data and use it to conduct sentiment analysis on the public's general trends, opinions and perceptions on Covid-19 protocols such as the mask mandate in NYC.


## Method
---

### Dataset introduction
- The dataset we used in our project was extracted from Twitter using the API provided by the Twitter Developer Account. The dataset includes 5 features: Text, Tag, Fav_count, Ret_count, and Time_created. The Text feature is the tweet text extracted by using a specific query. The Tag feature contains the labels that are annotated by the human annotators. The “1”s in the Tag section represent a “Positive” sentiment, whereas the “0”s in the Tag section represent “Negative” sentiment. Those labels are used as the Gold Standard for us to train and test the classifier models later. The Fav_count stands for the favorite count, i.e. how many likes this tweet has. The favorite count is not very useful when conducting the sentiment analysis on tweets text data. However, they can be used to do further data analysis on how the favorite counts affect the result of the labeling. Next, the Ret_count stands for the retweet count, which was not important in this study. The Time_created represents the time the tweet was created. The times that are included in this feature are formatted in UTF – 8 frames in python.

- In the query of the tweet extraction application, we use # Mask and # NYC as our query words to search for tweets that have those two hashtags. Those are the tweets that are highly relevant to our research topic. In addition to the words used in the query, we also limited the language to English only. This will exclude non-english tweets since English is the only language that all 3 annotators can speak and write. We use the “extended” mode to search for tweets since this mode will include all the matched tweets instead of limiting to only “short” tweets. The time frame of the extraction is from 3-25 to 3-28, which provides us enough tweets to annotate and train the model while ensuring a good amount of work in a limited time.

- More importantly, the retweets are excluded due to certain reasons. When writing a retweet to a specific tweet, users tend to read the initial tweet before writing down their own contents. The contents in the original tweet will then largely affect the objectivity of the retweet. Therefore, to make sure the data is comprehensive and objective, the investigators decide to exclude the  retweets and include original tweets only.

### Data annotation
* Data annotation is the process of labeling information that is available in a variety of formats such as text, video, and photographs. In our data set we annotated 200 tweets using binary labeling (1 for positive tweets and 0 for negative tweets) made by two annotators. An overall inter-annotator agreement by  85% was achieved and a third annotator addressed the conflict between the unsimilar tagging.  Labeled data sets are essential for supervised machine learning in order for the machine learning algorithm to be able to recognize and interpret the input patterns quickly and clearly. Even for humans, interpreting a viewpoint conveyed in a brief sentence, particularly when taken from social media posts and without any context, is a difficult task. Users may not adhere to appropriate spelling, syntax, and grammar norms, but they do tend to highlight words and emotions in accordance with the way they like others to understand their postings. In reading a written post, interpreting the emotion communicated is difficult since focusing solely on the content of the words might lead to misunderstanding of the text. Annotation of social media-related information, on the other hand, may be difficult.


### Browsing the dataset
* Based on the result of the human annotators, the dataset contains 49 positive sentiment tweets and 46 negative sentiment text. The distribution of the two classes are shown below. As the graph shows, the two classes are distributed evenly with a 6.12% difference for the positive class and a 6.52 % difference for the negative class. As a result, there’s no need to conduct any up or downsampling to balance the distribution of the two classes before we split the data into training and testing sets.

![](https://i.imgur.com/VxsaPO0.png) [Class Distribution]

* In addition to the distribution of sentiments, the investigators also created the Word Cloud using the Word Cloud package in python. The investigators created a Word Cloud for the text data in each of the sentiment and all the general text data. As we can see in all three of the graphs, the words “NYC”, “Mask”, and “Mandate” appear most frequently. In positive class, the words “Public” and “Still” happened more frequently than in negative and general classes . Moreover, in negative class, the words “Subway” and “Athlete” show up more frequently than in positive and general classes.


### Preprocess the data

* The text data are extracted from the first column of the dataset, which is named “Text”. Then, the investigators use the Spacy NLP pipeline to tokenize the data and vectorize them into matrices that contain numeric values. The numeric values in the matrix are used as the “X” values. The labels from the “Tag” section are also extracted and used as the standard classification result for further training in different classifiers.

* To examine the overall distribution of the vectorized text data, the PCA and tSNE plot are built. The PCA model from sklearn.decomposition is implemented and used to fit on the vectorized text data. The shape of the output X_pca is (95, 2). The datapoints are then being plotted in the one PCA plot. The blue dots show the distribution of negative sentiment text, whereas the red dots represent the positive sentiment text. The 2 principle components, PC1 and PC2, represent the most of the variance in our dataset.

![](https://i.imgur.com/CHF1duH.png) [PCA]

* In addition to the PCA plot, we also built the tSNE plot to dig further on the overall distribution of the text data. Compared to the PCA plot, the tSNE plot is being used to reduce the dimension of the data into 2D structure. Since we are using only the “Text” feature to fit the model and perform the classification on sentiments, the tSNE plot is not a very good representation of the distribution of the data.

![](https://i.imgur.com/9oIi1Yf.png) [tSNE]

### Data Splitting

* After having the vectorized data, we further splitted the data into training and testing sets. The model we used to perform the splitting is the train_test_split model from sklearn.model_selection. The training set and testing set were split into 80% and 20% with no random state. The newly generated matrix X_train had a shape of (76, 300); the matrix X_test had a shape of (76, ); the matrix y_train had a shape of (19, 300); and finally the matrix y_test had a shape of (19, ).


### Fit the classifiers with preprocessed data

* In the experiment design, the investigators planned to use four different classifiers to predict the sentiment tags of the text. The classifiers include: BernoulliNB, RandomForest, SVC, and K means. For the BernoulliNB model, we calculated its cross-validation score, the overall mean accuracy of the prediction, the Precision, Recall, F1 score. We also made the ROC curve for this model and calculated its AUC (Area Under Curve). In the sklearn RandomForest classifier we defined a max depth of 2 and a random state of 10. We also calculated its mean accuracy and the confusion matrix. The ROC curve was plotted and the AUC was calculated for this model as well. In the SVC classifier from sklearn.svm, we calculated the cross-validation score and the mean accuracy of the model. The confusion matrix was built and the Precision, Recall, and F1 score were calculated. In the KNN classifier, we first test the most appropriate number of clusters by using a range of K = 2-10. The Sum of squared distances from samples to closest clusters and the Silhouette score were recorded and plotted as the K increased from 2 to 10. Then, we used the most appropriate number of clusters to fit the KNN classifier and calculated its confusion matrix. The resulting clusters were visualized in a 2D graph, and the ROC curve was plotted with its AUC calculated. In addition, the resulting clusters were plotted using the tSNE graph to show the overall distribution of the datapoints in two different classes.


## Result
---

### BernoulliNB

* The BernoulliNB classifier from the naïve_bayes models in sklearn was trained and tested in our experiment. The cross-validation scores for the classifier for the entire dataset (X, y) were [0.68421053, 0.42105263, 0.42105263, 0.52631579, 0.63157895]. These relatively low scores showed that our classifier was underfitting and we might need more data to train the classifier to improve its performance. The mean accuracy of the BernoulliNB classifier using the test set was 0.7895. Based on the sentiment prediction (y_pred) and the annotated sentiments in the test set (y_test), the Precision, Recall, and F1 score were calculated. The Precision was 0.8889, the Recall was 0.7273, and the F1 score was 0.8000. The ROC curve was also plotted based on the False Positive Rate and True Positive Rate. The Area Under Curve (AUC) was calculated as 0.7727.

![](https://i.imgur.com/eJgD15k.png)


### RandomForest

* The RandomForestClassifier from sklearn.ensemble was used to predict the sentiments based on the text data and annotated tags. The investigators defined the max depth of the RandomForest as 2 with a random state of 10. After defining the hyperparameters, the RandomForest classifier was trained based on the data in the training set. The mean accuracy was calculated after the training: 0.5789. The Precision of the confusion matrix was 0.6667; the Recall was calculated as 0.5455; the F1 score was 0.6000. The overall performance of this model was not as good as we expected. The ROC curve was then plotted, and the AUC was calculated as 0.6818.

![](https://i.imgur.com/ujxdsvJ.png)


### SVC

* The SVC classifier from sklearn.svm was also used to perform the sentiment analysis on the twitter data we extracted. The overall performance was evaluated by using cross-validation scores and confusion matrix. The cross-validation scores were [0.52631579, 0.47368421, 0.52631579, 0.68421053, 0.52631579]. The mean accuracy of the model was 0.4737, the Precision was 0.5556, the Recall was 0.4545, and the F1 score was 0.5000. The overall performance, based on the evaluation from different perspectives, was not as good as we expected. Thus, we didn’t investigate further on this model and did not generate the ROC and AUC for the evaluation purpose.


### KNN

* The KMeans clustering model was imported from the sklearn.cluster package. First of all, the investigators used a range of K (from 2 to 10 ) to see how the Sum of squared distances from samples to closest clusters and Silhouette score changed while increasing the number of clusters. The scores were plotted, and the slope was evaluated to determine the most suitable number of clusters for the KNN classifier.

![](https://i.imgur.com/d5lcPpK.png)

* The slope on the left graph indicates that the Sum of squared distances from samples to closest clusters will not change its decreasing rate when having an increasing number of predefined clusters. Thus, the initial value 2 will be the most appropriate number of clusters. Also, based on our Gold Standard, the sentiment was annotated into 2 classes: Positive and Negative. These two pieces of evidence allowed the researchers to identify 2 as the number of predefined clusters.

* By using 2 clusters to perform classification, the KNN got a Precision of 0.5294, a Recall of 0.8182, and an F1 score of 0.6429. To better understand the accuracy and performance of the KNN classifier, we further visualized the two resulting clusters in a 2D graph and calculated the AUC: 0.4091.

![](https://i.imgur.com/DjBT7EW.png)

![](https://i.imgur.com/AEnHaLD.png)

* To better understand the separation of the datapoints in the two resulting clusters, we used a 2D tSNE graph to represent the two predicted sentiments. The “Cluster 0” represents the Negative sentiment datapoints; the “Cluster 1” represents the Positive sentiment datapoints. As we can see in the graph, the two classes of sentiments were clearly separated based on a horizontal line in the middle of the two clusters.

![](https://i.imgur.com/AIbI2Mg.png) [tSNE for KNN]

## Limitations
---

* There are several limitations to our project. In the first instance, the size of the dataset had a direct impact on the performance of our algorithms. It is possible that increasing the amount of training data will enhance accuracy 3. In our dataset, there are only 95 annotated records, which only gives the classifiers a very limited amount of training data. In another study conducted in 2021, the researchers collected the unique tweets from March 11, 2020 (the date that World Health Organization declared the COVID-19 pandemic) to January 31, 2021. Their dataset contains 1,499,421 unique tweets from 583,499 different users. Their prediction model successfully extracted a Positive trend on the public acceptance of the COVID-19 vaccines in November 2020 when Pfizer announced that their vaccines have 90% efficiency 4. Based on the comparison of the two studies, we are aware that in order to reduce the estimation variance and have better predictive performance, it is critical to extract and annotate more Twitter text data from a broader time frame to fit the classifiers before we conduct any further sentiment analysis 5.

* In addition, employing a more extensive annotation method for sentiment analysis may provide additional insight into the users' actions and thoughts. In our experiment design, we used the text data only to fit the classifiers to make the binary classification on the tags (Positive or Negative). In an existing sentiment analysis conducted in 2021 that used Twitter data to predict the sentiment and emotion of the users towards the usage of COVID-19 vaccines,  the researchers used the AFINN lexicon that provided them a sentiment range from -5 (highly Negative) to +5 (highly Positive) 6. Compared to our study, a more specific and broader range of sentiments will possibly provide more accurate results on different emotions since people tend to have a more complex emotion than “like” or “not like” when posting their tweets. Furthermore, to conduct a comprehensive sentiment analysis, we also need to use other features, such as the creation data of the tweets, the location of the users, and the retweet and favorite count, to fit the classifier and have a deeper training process. Those features are very useful when we need to do further analysis on our dataset. For instance, the location feature may suggest that people who live in different states in the U.S. or different districts in New York City tend to have a different average sentiment toward certain public health policies. The favorite and retweet counts will also imply the general acceptance of the original tweet.

* Moreover, the interpretation of tweets that contain sarcasm, jokes and emojis is a difficult task because users tend to highlight words and emotions in accordance with the way they like others to understand 7. This makes it difficult for the human annotator to understand. However, to make better sentiment analysis on how the general public react to a new public health emergency or a new healthcare policy, it is important to include the usage of emoji in the study. In a recent study that was published in March 2022, the investigators found that the normal pattern of emoji usage on social networks tends to change significantly following the happening of a public emergency in China. This change in emoji usage could largely reveal the knowledge gap of online behavioral changes and the evolution of the sentiments when a new outbreak takes place 8. To capture the potential sentiment that are expressed by the emoji and other symbols on social media such as Twitter, we need to find better classifiers that can take the emoji as an input value and combine this new feature with other independent variables to make the prediction on sentiment.


## Conclusion
---

* Sentiment analysis on social media platforms is an effective method to assess public opinions with rising public health trends. COVID-19 presented a lot of challenging public health issues that have been studied and helped in the understanding of people's perceptions and opinions. People's reactions and perception of emerging public health guidelines and policies should be studied and monitored in order to assess the policies compliance and develop appropriate measures to raise awareness and improve people compliance. Currently, policies related to the COVID-19 pandemic, such as Mask mandate require strict society compliance to achieve its aim. We believe that understanding society's perception will help policymakers and government leaders raise public awareness and assess the people's compliance. Moreover, Twitter provides real-time and accessible data to make such analysis and it's widely used among society, making it a valuable tool in doing public health research. This data needs to be analyzed and annotated properly to produce valuable insights.


## Reference
---

1.	Ji X, Chun SA, Wei Z, Geller J. Twitter sentiment classification for measuring public health concerns. Soc Netw Anal Min. 2015;5(1):13. doi:10.1007/s13278-015-0253-5

2.	Zunic A, Corcoran P, Spasic I. Sentiment Analysis in Health and Well-Being: Systematic Review. JMIR Med Inform. 2020;8(1):e16023. Published 2020 Jan 28. doi:10.2196/16023

3.	Shofiya C, Abidi S. Sentiment Analysis on COVID-19-Related Social Distancing in Canada Using Twitter Data. Int J Environ Res Public Health. 2021;18(11):5993. Published 2021 Jun 3. doi:10.3390/ijerph18115993

4.	Lyu JC, Han EL, Luli GK. COVID-19 Vaccine-Related Discussion on Twitter: Topic Modeling and Sentiment Analysis. J Med Internet Res. 2021;23(6):e24435. Published 2021 Jun 29. doi:10.2196/24435

5.	CHAWLA V. Is More Data Always Better For Building Analytics Models? OPINIONS. 2020; Available on: https://analyticsindiamag.com/is-more-data-always-better-for-building-analytics-models/

6.	Marcec R, Likic R. Using Twitter for sentiment analysis towards AstraZeneca/Oxford, Pfizer/BioNTech and Moderna COVID-19 vaccines [published online ahead of print, 2021 Aug 9]. Postgrad Med J. 2021;postgradmedj-2021-140685. doi:10.1136/postgradmedj-2021-140685

7.	Rehbein, Ines et al. Discussing best practices for the annotation of Twitter microtext. Sofia: Bulgarian Academy of Sciences, 2013. Pp. 73-84.

8.	Liu C, Tan X, Zhou T, Zhang W, Liu J, Lu X. Emoji use in China: popularity patterns and changes due to COVID-19 [published online ahead of print, 2022 Mar 21]. Appl Intell (Dordr). 2022;1-11. doi:10.1007/s10489-022-03195-y
