from markupsafe import string
import pandas as pd
import string
from gensim import corpora
from wordcloud import STOPWORDS,WordCloud
import matplotlib.pyplot as plt

from clean_tweets_dataframe import Clean_Tweets

tweeter_data = pd.read_csv("data\processed_tweet_data.csv")


class DataPreprocessor:



    def __init__(self):
        self.tweeter_data = tweeter_data
        

    def clean_data(self):
        Clean_Tweets.drop_unwanted_column(self,df=self.tweeter_data)
        Clean_Tweets.drop_duplicate(self,df=self.tweeter_data)
        Clean_Tweets.remove_non_english_tweets(self,df=self.tweeter_data)

        

        return self.tweeter_data
    
    def preprocess_data(self):

        self.tweeter_data['original_text'] = self.tweeter_data['original_text'].astype(str)
        self.tweeter_data['original_text'] = self.tweeter_data['original_text'].apply(lambda x: x.lower())
        self.tweeter_data['original_text']= self.tweeter_data['original_text'].apply(lambda x: x.translate(str.maketrans(' ', ' ', string.punctuation)))


        sentence_list = [tweet for tweet in self.tweeter_data['original_text']]
        word_list = [sent.split() for sent in sentence_list]
        word_to_id = corpora.Dictionary(word_list)
        corpus = [word_to_id.doc2bow(tweet) for tweet in word_list] 


        return word_to_id, corpus

    def explore_data(self):
        first_ten_values = self.tweeter_data.head()
        shape_of_data = self.tweeter_data.shape
    
        return first_ten_values, shape_of_data
    
    def visualize_data(self):
        plt.figure(figsize=(20, 10))
        plt.imshow(WordCloud(width=1000,height=600,stopwords=STOPWORDS).generate(' '.join(self.tweeter_data.original_text .values)))
        plt.axis('off')
        plt.title('Most Frequent Words In Our Tweets',fontsize=16)
        plt.show()


if __name__ == "__main__":
    test= DataPreprocessor()
    #print(test.clean_data().shape)
    #print(test.preprocess_data())
    #print(test.explore_data())
    #print(test.visualize_data())


