# Twitter Sentiment Analysis on Mask Mandate Policy in NYC
###### **Author: Abdullah Altammami; Jing Gao; Nathanael George**
###### **Weill Cornell Graduate School of Medical Sciences, New York, United States**

## 1. How to extract tweets with specific queries?
* In `Extract_tweets.ipynb`:
    * Set the API keys and secret (from Twitter Developer Portal)
    * Set the Access keys and secret (from Twitter Developer Portal)

* In `hashtag_tweets = tweepy.Cursor(api.search, q=["Mask", "NYC"], lang = "en", tweet_mode='extended').items(3000)`:
    * `q=[]` will set the specific queries. For example, the "Mask" and "NYC" are the example queries that we used to extract tweets. All the tweets containing those two tags will be extracted from Twitter.
    * `lang="en"` set the language to English. All the related tweets in English will be extracted. 
    * `tweet_mode="extended"` set the search mode so that all the tweets (including short and long tweets) will be included.
    * `item(3000)` specifies that we want to extract 3000 related tweets.

* The following codes will exclude all the retweets since we believe that they are not objective enough to represent the public opinions. However, if you want to keep those retweets, you can delete or comment out the `#Filter the tweets` part and append all the text into the text list. 

* In the **Exporting** part, we exported the data into the excel format using `data.to_excel()` function. However, you can also convert it to different formats, like csv files, using different function (`data.to_csv()`). 


## 2. How to train and evaluate the classifiers?
