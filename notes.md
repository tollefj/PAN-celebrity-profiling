# pan celebrity profiling
a project in tdt4310

## data management
the original data was provided as `.ndjson` files, with the raw text (feeds.ndjson) and labels (labels.ndjson).

Due to its size, we decided to work with the project using only the first 2000 feeds (or users), with multiple texts for each one.

An example from _one_ user (with *N* tweets) could look like this:

```
{
	"text":[
		"tweet_text1",
		"tweet_text2",
		...,
		"tweet_textN"
		], 
	"id": 12345
}
```

In the `labels.ndjson` file, we find the matching profile information for a given id. The textual data has to be matched with this information before any processing can be made.

We decided to start by merging the two files, grouping them on the id. All processing of the data has been done with [pandas](https://pandas.pydata.org/).

```python
feeds = pd.read_json('./feeds.ndjson', lines=True)
labels = pd.read_json('./labels.ndjson', lines=True)
```
Merging:
`combined = pd.merge(feeds, labels, on='id')`

We now have a dataframe with all needed data. The next step is dealing with the list of texts. For this, we unpack the list as a stack, keeping the id in place:

```python
expanded = combined.set_index(
    ['id', 'birthyear',
	'fame', 'gender', 'occupation']
)['text'].apply(pd.Series).stack()
expanded = expanded.reset_index()
expanded = expanded.drop(columns=['level_5'])
# level_5 is the auto-generated new column, 
# containing an index created during the
# stack()-operation.

```

This results in the following dataframe:

|#|birthyear|fame|gender|occupation	|text|
|---|---|---|---|---|---|
|0|1991|star|male|performer|...|
|1|1991|star|male|performer|...|
|2|1991|star|male|performer|...|
|3|1991|star|male|performer|...|
|...|...|...|...|...|...|

## data preprocessing

We now want to clean the text files before continuing, this is done with several functions, applied using the `.apply` function for the `'text'` row of the dataframe. These mostly use regex to clean punctuation, mentions in tweets, and other data we deemed irrelevant before letting the machine scour through the data.

NLTK provides us with stopwords and a stemmer. The stemmer (snowballstemmer: http://www.nltk.org/howto/stem.html) is used so we can hopefully find more similarities between texts (making sure the algorithms don't differ on simple plurals etc.)

We split the data 80:20 (train:test) using sklearn's `train_test_split`, and continued by creating a vectorizer ([TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html))to fit all textual data on, before transforming. 

## classification
...