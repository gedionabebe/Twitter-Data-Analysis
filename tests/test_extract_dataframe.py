import unittest
import pandas as pd
import sys, os

sys.path.append(os.path.abspath(os.path.join("../..")))

from extract_dataframe import read_json
from extract_dataframe import TweetDfExtractor

# For unit testing the data reading and processing codes, 
# we will need about 5 tweet samples. 
# Create a sample not more than 10 tweets and place it in a json file.
# Provide the path to the samples tweets file you created below
sampletweetsjsonfile = "global_twitter_data.json"    
_, tweet_list = read_json(sampletweetsjsonfile)

columns = [
    "created_at",
    "source",
    "original_text",
    "clean_text",
    "sentiment",
    "polarity",
    "subjectivity",
    "lang",
    "favorite_count",
    "retweet_count",
    "original_author",
    "screen_count",
    "followers_count",
    "friends_count",
    "possibly_sensitive",
    "hashtags",
    "user_mentions",
    "place",
    "place_coord_boundaries",
]


class TestTweetDfExtractor(unittest.TestCase):
    """
		A class for unit-testing function in the fix_clean_tweets_dataframe.py file

		Args:
        -----
			unittest.TestCase this allows the new class to inherit
			from the unittest module
	"""

    def setUp(self) -> pd.DataFrame:
        self.df = TweetDfExtractor(tweet_list[:5])
        self.tweet_df = self.df.get_tweet_df()

    def test_find_statuses_count(self):
        self.assertEqual(
            self.df.find_statuses_count(), list(self.tweet_df.head()['statuses_count'])
        )

    def test_find_full_text(self):
        text = self.tweet_df.head()['original_text']

        self.assertEqual(self.df.find_full_text(), list(text))
        

    def test_find_sentiments(self):
        self.assertEqual(
            self.df.find_sentiments(self.df.find_full_text()),
            (
                list(self.tweet_df.head()['polarity']),
                list(self.tweet_df.head()['subjectivity']),
                list(self.tweet_df.head()['sentiment']),
            ),
        )


    def test_find_screen_name(self):
        name = self.tweet_df.head()['original_author']
        self.assertEqual(self.df.find_screen_name(), list(name))

    def test_find_followers_count(self):
        f_count = self.tweet_df.head()['followers_count']
        self.assertEqual(self.df.find_followers_count(), list(f_count))

    def test_find_friends_count(self):
        friends_count = self.tweet_df.head()['friends_count']
        self.assertEqual(self.df.find_friends_count(), list(friends_count))

    def test_find_is_sensitive(self):
        self.assertEqual(self.df.is_sensitive(), list(self.tweet_df.head()['possibly_sensitive']))


    def test_find_hashtags(self):
         self.assertEqual(self.df.find_hashtags(), list(self.tweet_df.head()['hashtags']))

    def test_find_mentions(self):
         self.assertEqual(self.df.find_mentions(), list(self.tweet_df.head()['user_mentions']))



if __name__ == "__main__":
    unittest.main()

