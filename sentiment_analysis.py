"""Amazon Review Sentiment Analysis

This script loads the amason_product_reviews.csv dataset,
selects and cleans the data in the reviews.text column,
before processing the reviews for NLP and sentiment analysis.

Script also defines a function to get the polarity of a given
review and categorise the sentiment into 'Positive', 'Neutral,
and 'Negative', and a function to compare 2 reviews using the
similarity method.

The script prints to the console the polarity and predicted
sentiment of 2 chosen reviews and their similarity.

This script requires the 'pandas' and 'spaCy' libraries
are installed in the current Python enviroment and
additionally that the SpacyTextBlob is imported.

"""

import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

df = pd.read_csv('/Users/44785/Desktop/CGRMR/Fiddling/Movies/amazon_product_reviews.csv')

# EDA revealed ~600 duplicates to be dropped along with na
reviews_data = df['reviews.text'].dropna().drop_duplicates()

# Reindex data to avoid KeyErrors later
reviews_data.reset_index(drop=True, inplace=True)

# Convert data to string, convert to lower case and strip whitespace.
reviews_clean = (
    reviews_data.astype('string')
    .str.strip()
    .str.lower()
    )
# Load spaCy model for NLP along with pipe for sentiment analysis
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

def preprocess(text):
    '''Performs NLP on text and returns a string of the joined,
    lemmatized tokens without stop words or punctuation.'''
    doc = nlp(text)
    return (' '.join(
            [token.lemma_ for token in doc 
            if not token.is_stop 
            and not token.is_punct])
            )

# Apply preprocess function to the set (~4400) of reviews (takes ~1m).
processed_reviews = reviews_clean.apply(preprocess)

def sentiment_analysis(review):
    '''Takes text and returns it's polarity
    and predicted sentiment as a tuple.'''
    doc = nlp(review)

    def get_polarity(doc):
        '''Returns polarity attribute of doc.'''
        return doc._.blob.polarity
    
    def predict_sentiment(polarity):
        '''Categorises polarity as "Positive" for values > 0.2, 
        "Negative" for values < -0.2., "Neutral" otherwise'''
        if polarity < -0.2:
            return 'Negative'
        elif polarity > 0.2:
            return 'Positive'
        else:
            return 'Neutral'
    
    polarity = get_polarity(doc)
    sentiment = predict_sentiment(polarity)

    return polarity, sentiment

# Using the 'sm' spaCy model reduces accuracy of 
# similarity values as word vectors are not loaded.
def sentiment_comparison(review_1, review_2):
    '''Takes 2 reviews and gets the similarity value.'''
    doc_1 = nlp(review_1)
    doc_2 = nlp(review_2)
    return doc_1.similarity(doc_2)

# Choose 2 reviews to test on.
while True:
    try:
        user_num1 = int(
            input(f"Enter an integer from 0 to {len(reviews_clean)}: "))
        if user_num1 < 0:
            raise ValueError
        test_r1 = reviews_clean[user_num1]
        break
    except KeyError:
        print("Index out of range")
    except ValueError:
        print("Please enter a valid integer")

while True:
    try:
        user_num2 = int(
            input(f"Enter another integer from 0 to {len(reviews_clean)}: "))
        if user_num1 < 0:
            raise ValueError
        test_r2 = reviews_clean[user_num2]
        break
    except KeyError:
        print("Index out of range")
    except ValueError:
        print("Please enter a valid integer")

# Print polarity, sentiment and similarity for test reviews.
print(f"Review {user_num1}: {test_r1}\n Polarity and predicted sentiment: {sentiment_analysis(test_r1)}")
print(f"Review {user_num2}: {test_r2}\n Polarity and predicted sentiment: {sentiment_analysis(test_r2)}")
print(f"Similarity score: {sentiment_comparison(test_r1, test_r2)}")