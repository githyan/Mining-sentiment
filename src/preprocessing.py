import emoji
import string
import nltk
from nltk.corpus import stopwords
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.model_selection import train_test_split

nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab")
stopwords_ind = set(stopwords.words("indonesian"))
# Add 'yg' and 'nya' to the stopword list as requested by the user
stopwords_ind.add("yg")
stopwords_ind.add("nya")
stopwords_ind.add("ya")


def preprocessing_text(text):
    if not isinstance(text, str):
        return []
    text = emoji.replace_emoji(text, replace="")
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word not in stopwords_ind]

    return filtered_words


def df_resamplings(dataframe):
    ros = RandomOverSampler(random_state=42)

    dataframe["cleaned_comments"] = dataframe["comments"].apply(preprocessing_text)

    X_resampled, y_resampled = ros.fit_resample(
        dataframe.drop(columns=["sentiments"], errors="ignore"), dataframe["sentiments"]
    )
    dataset_resampled = X_resampled.copy()
    dataset_resampled["sentiments"] = y_resampled

    dataset_resampled = dataset_resampled.sample(frac=1, random_state=42).reset_index(
        drop=True
    )
    return dataset_resampled


def drop_emptylist(dataframe):
    dataframe_resampled = df_resamplings(dataframe)
    masking = dataframe_resampled["cleaned_comments"].apply(lambda x: len(x) > 0)
    dataframe_resampled["cleaned_comments"] = dataframe_resampled[
        "cleaned_comments"
    ].apply(lambda x: " ".join(x))
    dataframe_resampled = dataframe_resampled[masking].copy()
    dataframe_resampled.reset_index(
        drop=True, inplace=True
    )  # Added: Reset index after filtering

    return dataframe_resampled


def splits():
    print("Hello from project-skripsi!")
    dataframe = pd.read_csv("vader_dataset_sentiment.csv")
    dataframe_resampled = drop_emptylist(dataframe)

    # Step 1: Split data into training (70%) and temporary (30%) sets
    X = dataframe_resampled["cleaned_comments"]
    y = dataframe_resampled["sentiments"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Step 2: Split the temporary set (30%) into validation (15%) and test (15%) sets
    # Since X_temp is 30% of the original data, 0.5 of X_temp will be 15% of the original data
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"X_train shape: {X_train.shape} y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape} y_test shape: {y_test.shape}")
    print(f"X_val shape: {X_val.shape} y_val shape: {y_val.shape}")
    # Create a mapping from sentiment labels to numerical values
    label_mapping = {"positive": 1, "neutral": 0, "negative": 2}

    # Apply the mapping to the sentiment labels
    y_train_numerical = y_train.map(label_mapping).values
    y_val_numerical = y_val.map(label_mapping).values
    y_test_numerical = y_test.map(label_mapping).values

    print("Numerical labels created successfully:")
    print(f"y_train_numerical shape: {y_train_numerical.shape}")
    print(f"y_val_numerical shape: {y_val_numerical.shape}")
    print(f"y_test_numerical shape: {y_test_numerical.shape}")
    print("Example of y_train_numerical:", y_train_numerical[:5])

    return X_train, X_val, X_test, y_train_numerical, y_val_numerical, y_test_numerical
