import tweepy
import json
import pickle
import os
import data_cleaner
from keras.models import load_model

print("Welcome to the Twitter Celebrity Profile Analyser!")
print("Enter usernames to examine, or 'exit' to exit.")


def auth_api(keyfile):
    with open(keyfile) as f:
        keys = json.load(f)

    auth = tweepy.OAuthHandler(keys["cons_key"], keys["cons_secret"])
    auth.set_access_token(keys["access_token"], keys["access_secret"])

    api = tweepy.API(auth)
    return api


def search_twitter(api, query, max_id=None):
    if max_id is None:
        results = api.user_timeline(screen_name=query, count=200, tweet_mode="extended")
    else:
        results = api.user_timeline(screen_name=query, count=200, tweet_mode="extended", max_id=max_id)
    tweets = []
    for result in results:
        text = result.full_text
        tweets.append(text)
    return tweets, results.max_id


def scrape_twitter_timeline(api, query):
    print("Scraping twitter timeline...")
    scraped_tweets = []
    tweets, max_id = search_twitter(api, query)
    scraped_tweets.extend(tweets)
    while (True):
        if len(tweets) < 100 or len(scraped_tweets) > 3000:
            return scraped_tweets
        else:
            tweets, max_id = search_twitter(api, query, max_id=max_id)
            print("Got " + str(len(tweets)) + " tweets, and a total of " + str(len(scraped_tweets)) + " thus far.")
            scraped_tweets.extend(tweets)


def predict_user(model, user_vec):
    result = model.predict([[user_vec]])
    fame = ["rising", "star", "superstar"]
    gender = ["female", "male", "nonbinary"]
    occupation = ["creator", "manager", "performer", "politics", "professional", "religious", "science", "sports"]

    birth_pred = result[0][0][0]
    fame_pred = result[1][0]
    gender_pred = result[2][0]
    occ_pred = result[3][0]

    _max = 2008
    _min = 1940
    year_pred = int(birth_pred * (_max - _min) + _min)
    print("Predicted birthyear: " + str(int(year_pred)))

    fame_pred = fame[fame_pred.argmax()]
    print("Predicted fame level: " + fame_pred)

    gend_pred = gender[gender_pred.argmax()]
    print("Predicted gender: " + gend_pred)

    occu_pred = occupation[occ_pred.argmax()]
    print("Predicted occupation: " + occu_pred)

def main():
    api = auth_api("keys.json")
    tokenizer = None
    model = None
    while (True):
        username = input("Username: ")
        if username == "exit":
            break
        tweets = scrape_twitter_timeline(api, username)
        print("Scraped " + str(len(tweets)) + " tweets from " + username)
        print("Cleaning dataset...")

        tweets = data_cleaner.clean(tweets)

        if tokenizer is None:
            print("Loading tokenizer...")
            try:
                with open(os.path.join('data', 'tokenizer.pickle'), 'rb') as handle:
                    tokenizer = pickle.load(handle)
            except FileNotFoundError:
                print("Could not find the tokenizer file! It must be named 'tokenizer.pickle' and reside in the 'data' folder!")
                continue
        print("Making tf-idf vector...")
        X = tokenizer.texts_to_matrix([tweets], mode='tfidf')
        tfidf_vec = X[0]
        if model is None:
            print("Loading ml model...")
            try:
                model = load_model(os.path.join("data", "model.h5"))
            except OSError:
                print("Could not find the model file! It must be named 'model.h5' and reside in the 'data' folder!")
                continue
        predict_user(model, tfidf_vec)

main()
