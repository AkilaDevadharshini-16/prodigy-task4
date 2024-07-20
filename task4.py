# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
dataset_path = 'twitter_sentiment.csv'  # Assuming the dataset is in the same directory as this script
df = pd.read_csv(dataset_path)

# Display the first few rows to understand the structure
print(df.head())

# Data Cleaning and Preprocessing
# Assuming 'text' column contains tweet text and 'sentiment' column contains sentiment labels

# Remove NaN values
df.dropna(subset=['text'], inplace=True)

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and links (basic cleaning)
    text = ' '.join(word for word in text.split() if not word.startswith('http'))
    return text

# Apply preprocessing to 'text' column
df['clean_text'] = df['text'].apply(preprocess_text)

# Sentiment Analysis using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Return polarity (range -1 to 1 where < 0 indicates negative sentiment, > 0 indicates positive sentiment)
    return analysis.sentiment.polarity

# Apply sentiment analysis
df['sentiment_score'] = df['clean_text'].apply(analyze_sentiment)

# Visualizations

# Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['sentiment_score'], bins=30, kde=True)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Word Clouds for each sentiment category
# Separate texts by sentiment
positive_texts = ' '.join(df[df['sentiment_score'] > 0]['clean_text'])
negative_texts = ' '.join(df[df['sentiment_score'] < 0]['clean_text'])
neutral_texts = ' '.join(df[df['sentiment_score'] == 0]['clean_text'])

# Function to generate word cloud
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# Generate word clouds
generate_word_cloud(positive_texts, 'Positive Sentiment Word Cloud')
generate_word_cloud(negative_texts, 'Negative Sentiment Word Cloud')
generate_word_cloud(neutral_texts, 'Neutral Sentiment Word Cloud')
