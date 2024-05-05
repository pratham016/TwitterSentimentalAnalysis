import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import emoji
import nltk
# import warnings
# from sklearn.exceptions import ConvergenceWarning

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=ConvergenceWarning)
# In[2]:


df = pd.read_csv('twcs.csv')

nltk.download('stopwords')

# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


# In[6]:


#Drop Unwanted Columns


# In[7]:


df.drop('created_at', axis=1, inplace=True)
df.drop('response_tweet_id', axis=1, inplace=True)
df.drop('in_response_to_tweet_id', axis=1, inplace=True)


# In[8]:


df.head()


# In[9]:


df = df[df['inbound'] != False]


# In[10]:


df.head()


# In[11]:


df.shape


# In[12]:


import string
from nltk.corpus import stopwords


# In[13]:


# convert all text to lowercase
df['text'] = df['text'].apply(lambda x: x.lower())


# In[14]:


# remove URLs
df['text'] = df['text'].apply(lambda x: re.sub(r'http\S+', '', x))


# In[15]:


# remove mentions and hashtags
df['text'] = df['text'].apply(lambda x: re.sub(r'@\w+|\#\w+', '', x))


# In[16]:


# remove punctuation and special characters
df['text'] = df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))


# In[17]:


df.head()


# In[18]:


df.shape


# In[19]:


# remove stop words
stop_words = set(stopwords.words('english'))
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))


# In[20]:


print(df)


# In[21]:


df.shape


# In[22]:


df.head()


# In[23]:


df.isna()


# In[24]:


#remove duplicates


# In[25]:


df.drop_duplicates(subset=['tweet_id'], inplace = True)


# In[26]:


df.shape


# In[27]:


df.dropna(subset=['text'], inplace=True)


# In[28]:


df.head(600)


# In[29]:


from nltk.stem import WordNetLemmatizer

# download the necessary resources for lemmatization
nltk.download('punkt')
nltk.download('wordnet')


# In[30]:




# In[31]:


# define a regular expression pattern to match emojis
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

# remove emojis from the 'text' column
df['text'] = df['text'].apply(lambda x: emoji_pattern.sub(r'', x))

# print the updated dataframe
print(df.head())


# In[32]:


df.shape


# In[33]:


# # create a lemmatizer object
# lemmatizer = WordNetLemmatizer()
# # lemmatize the 'text' column
# twcs['lemmatized_text'] = twcs['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x)]))


# In[34]:


df.to_csv('mini_processed.csv', index = False)


# In[35]:


df.shape


# In[36]:


df = pd.read_csv('mini_processed.csv')


# In[37]:


df.shape


# In[38]:


import nltk
nltk.download('vader_lexicon')


# In[39]:


import string
from collections import Counter

cnt = Counter()

for text in df['text'].values:
    if isinstance(text, str):  # check if text is a string
        for word in text.split():
            cnt[word] += 1

cnt.most_common(10)


# In[40]:


freq_words = set([w for (w,wc) in cnt.most_common(10)])

def remove_freqwords(text):
    return ' '.join(word for word in str(text).split() if word not in freq_words)

df['text'].apply(lambda text:remove_freqwords(text))
df.head()


# In[41]:


n_rare_words = 10
rare_words = set([w for (w,c) in cnt.most_common()][:- n_rare_words: -1])
print(rare_words)

def remove_rarewords(text):
    return ' '.join(word for word in str(text).split() if word not in rare_words)

df['text'].apply(lambda text:remove_rarewords(text))
df.head()


# In[42]:


df.shape


# In[43]:


from nltk.stem import WordNetLemmatizer
import nltk

# Define a function to perform lemmatization using the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


df['text'].apply(lambda x: lemmatize_words(x) if isinstance(x, str) else x)


# In[44]:


nltk.download('averaged_perceptron_tagger')


# In[45]:


text = "We are meeting tomorrow for our business dealings and paperwork signing."
pos_tagged_text = nltk.pos_tag(text.split())
print(pos_tagged_text)


# In[46]:


df.head()


# In[47]:


from nltk.stem import WordNetLemmatizer
import nltk

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    if isinstance(text, str):
        return ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text)])
    else:
        return text

df['text'] = df['text'].apply(lemmatize_text)


# In[48]:


df.head()


# In[49]:


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


# In[50]:


text = "Driverless AI NLP blog post on https://www.h2o.ai/blog/detecting-sarcasm-is-difficult-but-ai-may-have-an-answer/"
remove_urls(text)


# In[51]:


# for removal of emails

def remove_email(text):
    email_pattern = re.compile(r'\S+@\S+\.\S+')
    return email_pattern.sub(r'', text)

text = "Want to know more: send us an email at amrit@cs.unm.edu"
remove_email(text)


# In[52]:


def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

text = """<div>
<h1> H2O</h1>
<p> AutoML</p>
<a href="https://www.h2o.ai/products/h2o-driverless-ai/"> Driverless AI</a>
</div>"""

print(remove_html(text))


# In[53]:


from nltk.sentiment import SentimentIntensityAnalyzer


# In[54]:


df.to_csv('mini_preprocessed.csv', index = False)


# In[55]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the twcs dataset that contains the lemmatized text
df = pd.read_csv('mini_preprocessed.csv')

# Define a function to get the sentiment score using NLTK's VADER
sid = SentimentIntensityAnalyzer()
def get_sentiment_score(text):
    sentiment_score = sid.polarity_scores(str(text))
    return sentiment_score['compound']

# Get the sentiment scores for each tweet using NLTK's VADER
df['sentiment_score'] = df['text'].apply(get_sentiment_score)

# Assign a sentiment label to each tweet
threshold = 0.5
df['sentiment_label'] = df['sentiment_score'].apply(lambda x: 'positive' if x >= threshold else 'negative' if x <= -threshold else 'neutral')


# In[56]:


df.head()


# In[57]:


mean_sentiment_score = df['sentiment_score'].mean()
print(mean_sentiment_score)


# In[58]:


counts = df['sentiment_label'].value_counts()

# calculate the percentage of instances for each sentiment
percentages = counts / len(df) * 100

# print the results
print('Positivity: {:.2f}%'.format(percentages['positive']))
print('Negativity: {:.2f}%'.format(percentages['negative']))
print('Neutral: {:.2f}%'.format(percentages['neutral']))


# In[59]:


null_counts = df.isnull().sum()
print(null_counts)


# In[60]:


df = df.dropna()

print(df)


# In[61]:


print(null_counts)


# In[62]:


df.dropna()


# In[63]:


null_counts = df.isnull().sum()
print(null_counts)


# In[64]:


print(null_counts)


# In[65]:


mean_sentiment_score = df['sentiment_score'].mean()
print(mean_sentiment_score)


# In[66]:


counts = df['sentiment_label'].value_counts()

# calculate the percentage of instances for each sentiment
percentages = counts / len(df) * 100

# print the results
print('Positivity: {:.2f}%'.format(percentages['positive']))
print('Negativity: {:.2f}%'.format(percentages['negative']))
print('Neutral: {:.2f}%'.format(percentages['neutral']))


# In[67]:


df.head()


# In[68]:


df.to_csv('sentimentscored.csv', index = False)


# In[69]:


df = pd.read_csv('sentimentscored.csv')


# In[70]:


mean_sentiment_score = df['sentiment_score'].mean()
print(mean_sentiment_score)


# In[71]:


counts = df['sentiment_label'].value_counts()

# calculate the percentage of instances for each sentiment
percentages = counts / len(df) * 100

# print the results
print('Positivity: {:.2f}%'.format(percentages['positive']))
print('Negativity: {:.2f}%'.format(percentages['negative']))
print('Neutral: {:.2f}%'.format(percentages['neutral']))


# In[72]:


df.shape


# In[73]:




# Load the DataFrame
df = pd.read_csv('sentimentscored.csv')

# Split the text column into a list of words and count the number of words in each row
df['word_count'] = df['text'].str.split().apply(len)

# Filter out rows where the number of words is 1
df = df[df['word_count'] > 1]

# Drop the word_count column
df.drop(columns=['word_count'], inplace=True)

# Save the cleaned DataFrame
df.to_csv('remsingletwcs.csv', index=False)


# In[74]:


df.shape


# In[75]:


counts = df['sentiment_label'].value_counts()

# calculate the percentage of instances for each sentiment
percentages = counts / len(df) * 100

# print the results
print('Positivity: {:.2f}%'.format(percentages['positive']))
print('Negativity: {:.2f}%'.format(percentages['negative']))
print('Neutral: {:.2f}%'.format(percentages['neutral']))


# In[76]:


# Load the DataFrame
df = pd.read_csv('remsingletwcs.csv')

# Count the number of samples in each sentiment category
positive_count = len(df[df['sentiment_label'] == 'positive'])
negative_count = len(df[df['sentiment_label'] == 'negative'])
neutral_count = len(df[df['sentiment_label'] == 'neutral'])

# Find the maximum count of any sentiment category
max_count = max(positive_count, negative_count, neutral_count)

# Calculate the number of samples to add to each sentiment category
num_to_add = int(0.3 * max_count) - positive_count
if num_to_add > 0:
    # Randomly select samples from the positive sentiment category and append them to the DataFrame
    additional_samples = df[df['sentiment_label'] == 'positive'].sample(n=num_to_add, replace=True)
    df = pd.concat([df, additional_samples])
    
num_to_add = int(0.3 * max_count) - negative_count
if num_to_add > 0:
    # Randomly select samples from the negative sentiment category and append them to the DataFrame
    additional_samples = df[df['sentiment_label'] == 'negative'].sample(n=num_to_add, replace=True)
    df = pd.concat([df, additional_samples])
    
num_to_add = int(0.3 * max_count) - neutral_count
if num_to_add > 0:
    # Randomly select samples from the neutral sentiment category and append them to the DataFrame
    additional_samples = df[df['sentiment_label'] == 'neutral'].sample(n=num_to_add, replace=True)
    df = pd.concat([df, additional_samples])

# Shuffle the DataFrame to randomize the order of the samples
df = df.sample(frac=1).reset_index(drop=True)

# Save the balanced DataFrame
df.to_csv('balanced_dataframe.csv', index=False)


# In[77]:


counts = df['sentiment_label'].value_counts()

# calculate the percentage of instances for each sentiment
percentages = counts / len(df) * 100

# print the results
print('Positivity: {:.2f}%'.format(percentages['positive']))
print('Negativity: {:.2f}%'.format(percentages['negative']))
print('Neutral: {:.2f}%'.format(percentages['neutral']))


# In[78]:


df = pd.read_csv('balanced_dataframe.csv')


# In[79]:


df.shape


# In[80]:


import pandas as pd
import random

# Load the DataFrame
df = pd.read_csv('balanced_dataframe.csv')

# Calculate the number of samples to add to each sentiment category
num_to_add = int(0.1875 * len(df)) - len(df[df['sentiment_label'] == 'positive'])
if num_to_add > 0:
    # Randomly select samples from the positive sentiment category and append them to the DataFrame
    additional_samples = df[df['sentiment_label'] == 'positive'].sample(n=num_to_add, replace=True)
    df = pd.concat([df, additional_samples])

num_to_add = int(0.1875 * len(df)) - len(df[df['sentiment_label'] == 'negative'])
if num_to_add > 0:
    # Randomly select samples from the negative sentiment category and append them to the DataFrame
    additional_samples = df[df['sentiment_label'] == 'negative'].sample(n=num_to_add, replace=True)
    df = pd.concat([df, additional_samples])

num_to_add = int(0.1875 * len(df)) - len(df[df['sentiment_label'] == 'neutral'])
if num_to_add > 0:
    # Randomly select samples from the neutral sentiment category and append them to the DataFrame
    additional_samples = df[df['sentiment_label'] == 'neutral'].sample(n=num_to_add, replace=True)
    df = pd.concat([df, additional_samples])

# Shuffle the DataFrame to randomize the order of the samples
df = df.sample(frac=1).reset_index(drop=True)

# Save the balanced DataFrame
df.to_csv('new_balanced_dataframe.csv', index=False)


# In[81]:


df = pd.read_csv('new_balanced_dataframe.csv')


# In[82]:


counts = df['sentiment_label'].value_counts()

# calculate the percentage of instances for each sentiment
percentages = counts / len(df) * 100

# print the results
print('Positivity: {:.2f}%'.format(percentages['positive']))
print('Negativity: {:.2f}%'.format(percentages['negative']))
print('Neutral: {:.2f}%'.format(percentages['neutral']))


# In[83]:


df.head(50)


# In[84]:



from sklearn.utils import resample

# Load the dataset
df = pd.read_csv('new_balanced_dataframe.csv')

# Separate the samples by sentiment class
positive = df[df['sentiment_label'] == 'positive']
negative = df[df['sentiment_label'] == 'negative']
neutral = df[df['sentiment_label'] == 'neutral']

# Determine the number of samples to add to each class
num_to_add = len(neutral) - len(positive) - len(negative)

# Oversample the minority classes
positive_oversampled = resample(positive, replace=True, n_samples=num_to_add//2)
negative_oversampled = resample(negative, replace=True, n_samples=num_to_add//2)

# Combine the oversampled minority classes with the original dataset
df_oversampled = pd.concat([positive_oversampled, negative_oversampled, neutral])

# Shuffle the dataset
df_oversampled = df_oversampled.sample(frac=1).reset_index(drop=True)

# Save the balanced dataset to a new CSV file
df_oversampled.to_csv('minibalanced.csv', index=False)


# In[85]:


df = pd.read_csv('minibalanced.csv')


# In[86]:


counts = df['sentiment_label'].value_counts()

# calculate the percentage of instances for each sentiment
percentages = counts / len(df) * 100

# print the results
print('Positivity: {:.2f}%'.format(percentages['positive']))
print('Negativity: {:.2f}%'.format(percentages['negative']))
print('Neutral: {:.2f}%'.format(percentages['neutral']))


# In[87]:


df.shape


# In[91]:


df.head()


# In[92]:


import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('minibalanced.csv')

# Separate the samples by sentiment class
positive = df[df['sentiment_label'] == 'positive']
negative = df[df['sentiment_label'] == 'negative']
neutral = df[df['sentiment_label'] == 'neutral']

# Calculate the number of samples to add or remove
num_positives_to_add = int(len(neutral) * 0.2)
num_negatives_to_add = int(len(neutral) * 0.2)
num_neutrals_to_remove = int((len(negative) + len(positive)) * 0.2)

# Oversample the positive and negative classes
if num_positives_to_add > 0:
    positive_oversampled = positive.sample(n=num_positives_to_add, replace=True, random_state=1)
else:
    positive_oversampled = pd.DataFrame()
if num_negatives_to_add > 0:
    negative_oversampled = negative.sample(n=num_negatives_to_add, replace=True, random_state=1)
else:
    negative_oversampled = pd.DataFrame()

# Undersample the neutral class
if num_neutrals_to_remove > 0:
    neutral_undersampled = neutral.sample(n=num_neutrals_to_remove, random_state=1)
else:
    neutral_undersampled = pd.DataFrame()

# Concatenate the oversampled and undersampled datasets
df_balanced = pd.concat([positive_oversampled, negative_oversampled, neutral,], axis=0)

# Shuffle the rows
df_balanced = df_balanced.sample(frac=1, random_state=1)

# Reset the index
df_balanced.reset_index(drop=True, inplace=True)

# Print the class distribution
print(df_balanced['sentiment_label'].value_counts(normalize=True))


# In[93]:


counts = df['sentiment_label'].value_counts()

# calculate the percentage of instances for each sentiment
percentages = counts / len(df) * 100

# print the results
print('Positivity: {:.2f}%'.format(percentages['positive']))
print('Negativity: {:.2f}%'.format(percentages['negative']))
print('Neutral: {:.2f}%'.format(percentages['neutral']))


# In[94]:


import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('minibalanced.csv')

# Separate the samples by sentiment class
positive = df[df['sentiment_label'] == 'positive']
negative = df[df['sentiment_label'] == 'negative']
neutral = df[df['sentiment_label'] == 'neutral']

# Calculate the number of samples to add or remove
num_positives_to_add = int(len(df) * 0.3) - len(positive)
num_negatives_to_add = int(len(df) * 0.3) - len(negative)
num_neutrals_to_add = int(len(df) * 0.4) - len(neutral)

# Oversample the positive, negative, and neutral classes
if num_positives_to_add > 0:
    positive_oversampled = positive.sample(n=num_positives_to_add, replace=True, random_state=1)
else:
    positive_oversampled = pd.DataFrame()
if num_negatives_to_add > 0:
    negative_oversampled = negative.sample(n=num_negatives_to_add, replace=True, random_state=1)
else:
    negative_oversampled = pd.DataFrame()
if num_neutrals_to_add > 0:
    neutral_oversampled = neutral.sample(n=num_neutrals_to_add, replace=True, random_state=1)
else:
    neutral_oversampled = pd.DataFrame()

# Combine the oversampled dataframes with the original dataframe
balanced_df = pd.concat([df, positive_oversampled, negative_oversampled, neutral_oversampled])

# Shuffle the rows of the balanced dataframe
balanced_df = balanced_df.sample(frac=1, random_state=1)

# Save the balanced dataset to a new csv file
balanced_df.to_csv('getbalancedalready_new.csv', index=False)


# In[95]:


import pandas as pd

# Load the balanced dataset
df = pd.read_csv('getbalancedalready_new.csv')

# Calculate the percentage of each sentiment class
positive_percent = len(df[df['sentiment_label'] == 'positive']) / len(df) * 100
neutral_percent = len(df[df['sentiment_label'] == 'neutral']) / len(df) * 100
negative_percent = len(df[df['sentiment_label'] == 'negative']) / len(df) * 100

# Print the percentages
print(f"Positive: {positive_percent:.2f}%")
print(f"Neutral: {neutral_percent:.2f}%")
print(f"Negative: {negative_percent:.2f}%")


# In[96]:


import pandas as pd
import numpy as np

# Load the original dataset
df = pd.read_csv('getbalancedalready_new.csv')

# Separate the data by sentiment label
positive_df = df[df['sentiment_label'] == 'positive']
neutral_df = df[df['sentiment_label'] == 'neutral']
negative_df = df[df['sentiment_label'] == 'negative']

# Calculate the number of samples to keep for each sentiment label
max_samples = min(len(positive_df), len(neutral_df), len(negative_df))
positive_samples = int(0.4 * max_samples)
neutral_samples = int(0.4 * max_samples)
negative_samples = int(0.4 * max_samples)

# Skew the data by randomly selecting the required number of samples for each sentiment label
positive_indices = np.random.choice(positive_df.index, size=positive_samples, replace=False)
neutral_indices = np.random.choice(neutral_df.index, size=neutral_samples, replace=False)
negative_indices = np.random.choice(negative_df.index, size=negative_samples, replace=False)

# Concatenate the selected samples into a single dataframe
selected_df = pd.concat([positive_df.loc[positive_indices], neutral_df.loc[neutral_indices], negative_df.loc[negative_indices]])

# Shuffle the dataframe
selected_df = selected_df.sample(frac=1).reset_index(drop=True)

# Save the skewed dataset to a new file
selected_df.to_csv('skewed_dataset.csv', index=False)


# In[97]:


df = pd.read_csv('skewed_dataset.csv')


# In[98]:


# Calculate the percentage of each sentiment class
positive_percent = len(df[df['sentiment_label'] == 'positive']) / len(df) * 100
neutral_percent = len(df[df['sentiment_label'] == 'neutral']) / len(df) * 100
negative_percent = len(df[df['sentiment_label'] == 'negative']) / len(df) * 100

# Print the percentages
print(f"Positive: {positive_percent:.2f}%")
print(f"Neutral: {neutral_percent:.2f}%")
print(f"Negative: {negative_percent:.2f}%")


# In[99]:


df.shape


# In[100]:


df.head()


# In[101]:


mean_sentiment_score = df['sentiment_score'].mean()
print(mean_sentiment_score)


# In[102]:


df_new = pd.read_csv('skewed_dataset.csv')


# In[104]:


# Split the dataset
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, df_new['sentiment_label'], test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split

X = df_new['text']
y = df_new['sentiment_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[106]:


df_new.head()


# In[107]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the TwCS dataset
df_new = pd.read_csv('skewed_dataset.csv')

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_new['text'], df_new['sentiment_label'], test_size=0.3, random_state=42)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a SVM classifier
svm = LinearSVC()
svm.fit(X_train_vec, y_train)

# Make predictions on the testing set
y_pred = svm.predict(X_test_vec)

# Evaluate the performance of the classifier
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[108]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[109]:


# Calculate the percentage of each sentiment class
positive_percent = len(df[df['sentiment_label'] == 'positive'])
neutral_percent = len(df[df['sentiment_label'] == 'neutral'])
negative_percent = len(df[df['sentiment_label'] == 'negative'])

# Print the percentages
print(f"Positive: {positive_percent:.2f}")
print(f"Neutral: {neutral_percent:.2f}")
print(f"Negative: {negative_percent:.2f}")


# In[ ]:


#Function to categorize sentiment label values


# In[ ]:


df.shape

with open('svm_model.pkl', 'wb') as file:  
    pickle.dump(svm,file)

