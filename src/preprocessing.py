import emoji
import string 
import nltk
from nltk.corpus import stopwords
from imblearn.over_sampling import RandomOverSampler


nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab')
stopwords_ind = set(stopwords.words('indonesian'))
# Add 'yg' and 'nya' to the stopword list as requested by the user
stopwords_ind.add('yg')
stopwords_ind.add('nya')
stopwords_ind.add('ya')

def preprocessing_text(text):
    if not isinstance(text, str):
        return []
    text = emoji.replace_emoji(text, replace='')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word not in stopwords_ind]

    return filtered_words

def df_resamplings(dataframe):
    ros = RandomOverSampler(random_state=42)

    dataframe['cleaned_comments'] = dataframe['comments'].apply(preprocessing_text)

    X_resampled, y_resampled  = ros.fit_resample(dataframe.drop(columns=['sentiments'], errors='ignore'), dataframe['sentiments'])
    dataset_resampled = X_resampled.copy()
    dataset_resampled['sentiments'] = y_resampled

    dataset_resampled = dataset_resampled.sample(frac=1, random_state=42).reset_index(drop=True)
    return dataset_resampled

def drop_emptylist(dataframe):
    dataframe_resampled = df_resamplings(dataframe)
    masking = dataframe_resampled['cleaned_comments'].apply(lambda x: len(x) > 0)
    dataframe_resampled['cleaned_comments'] = dataframe_resampled['cleaned_comments'].apply(lambda x: ' '.join(x))
    dataframe_resampled = dataframe_resampled[masking].copy()
    dataframe_resampled.reset_index(drop=True, inplace=True) # Added: Reset index after filtering

    return dataframe_resampled

def splits():
    pass

