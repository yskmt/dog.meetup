"""
tweepy
http://tweepy.readthedocs.org/en/v3.2.0/getting_started.html

"""
import tweepy
from authos import tw0, tw1, tw2, tw3


auth = tweepy.OAuthHandler(tw0, tw1)
auth.set_access_token(tw2, tw3)

api = tweepy.API(auth)

geo = '37.426233,-122.141195,1mi'
for tweet in tweepy.Cursor(api.search,
                           q="#dog",
                           geolocation=geo,
                           count=10,
                           until="2015-07-19",
                           result_type="recent",
                           include_entities=True,
                           lang="en").items(1000):

    # print tweet.text
    # if tweet.coordinates is not None:
    print tweet.created_at, tweet.text
    print tweet.coordinates
 
def prn(x):
    print x

map(prn, [dt.text for dt in dog_tweets])



population = [10, 20, 30, 40, 50, 20]
income = [100, 1000, 2000, 4000, 10000, 20000]
