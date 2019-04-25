# pan celebrity profiling
a project in tdt4310

## data management
the original data was provided as `.ndjson` files, with the raw text (feeds.ndjson) and labels (labels.ndjson).

The size of the original feeds.ndjson file is almost 8 GB. We split the files into chunks of 4000 lines using the unix-command `split`: `split -dl 1500 --additional-suffix=.ndjson feeds.ndjson feed`, which results in 23 files of approximately 350 MB each (`feed00.ndjson`->`feed22.ndjson`). This lets us work with as many files as we desire. There's going to be a lot of data written to memory no matter what (storing as dataframe, etc., so having the option is great!

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


This lets us handle each text as an _X_ set, with the labels _birthyear, fame, gender_ and _occupation_ as labels in a _y_ set.

## data preprocessing

We now want to clean the text files before continuing, this is done with several functions, applied using the `.apply` function for the `'text'` row of the dataframe. These mostly use regex to clean punctuation, mentions in tweets, and other data we deemed irrelevant before letting the machine scour through the data.

NLTK provides us with stopwords and a stemmer. The stemmer (snowballstemmer: http://www.nltk.org/howto/stem.html) is used so we can hopefully find more similarities between texts (making sure the algorithms don't differ on simple plurals etc.)

We split the data 80:20 (train:test) using sklearn's `train_test_split`, and continued by creating a vectorizer ([TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html))to fit all textual data on, before transforming. 

## classification
handling multi-label data can be tricky, especially when it includes a big set of different labels, making the network deep. Label powerset, while a solid classification method for this kind of problem, it is not feasible due to the sheer size of unique labels, especially in the "birthyear" column. The "occupation" column also includes 7 different labels.

first 1000 lines picked from the already partitioned file. This resulted in data we could manage, given the hardware available. Total: 1.6GB


The idea:

We have lots of text! Handling multiple labels can be tricky when we're limited by hardware (examples, even down to 1 GB files, resulted in memoryerrors when using multi-label classification methods, like the ones found in scikit-multilearn). 

Let's create new datasets, with text corresponding to each label's values:

categories
|- birthyear (model 1)
|-- 1950 (a entries)
|-- 1951 (b entries, etc)
|-- ...
|-- 1995
|- fame (model 2)
|-- star
|-- superstar
|-- ...
|-- artist
|- gender (model 3)
|-- female
|-- male
|-- non-binary
|- occupation (model 4)
|-- sports
|-- actor
|-- ...
|-- singer

This way, each model will represent the highest value for each category!


https://keras.io/models/sequential/
steps_per_epoch: Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.

-> https://github.com/keras-team/keras/issues/2708#issuecomment-218781778

