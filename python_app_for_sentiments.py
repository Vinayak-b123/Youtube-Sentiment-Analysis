import re
from googleapiclient.discovery import build
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import nltk

LEMMATIZER = WordNetLemmatizer()
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


YT_API_KEY = "AIzaSyDgXf3WCOG-zkQlV9lPQsDS7qM1Oc2RabQ" 

def preprocess_comment(comment):
    try:
        comment = str(comment).lower().strip()
        comment = re.sub(r'http\S+|www\S+', '', comment)       # URLs
        comment = re.sub(r'[\n\r\t]', ' ', comment)            # newlines/tabs
        comment = re.sub(r'[^\x00-\x7F]+', ' ', comment)       # non-ASCII
        comment = re.sub(r'\s+', ' ', comment).strip()          # normalise spaces
        # Lemmatize only — no stopword removal (vectorizer vocab has stopwords)
        comment = ' '.join([LEMMATIZER.lemmatize(w) for w in comment.split()])
        return comment
    except Exception as e:
        print(f"Error preprocessing comment: {e}")
        return str(comment)
    
    
def load_model(model_path, vectorizer_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        return model, vectorizer

    except Exception:
        raise



model, vectorizer = load_model("./yt_senti.pkl", "./vectorizer.pkl")



def extract_video_id(url: str) -> str:
    """Pull the video ID out of any standard YouTube URL format."""
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",       # ?v=xxxx
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})", # youtu.be/xxxx
        r"(?:embed/)([a-zA-Z0-9_-]{11})",     # /embed/xxxx
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not find a valid video ID in the URL you entered.")



def scrape_comments(video_id: str, max_comments: int = 100) -> list[str]:
    """Use the YouTube Data API to fetch up to max_comments top-level comments."""
    youtube = build("youtube", "v3", developerKey=YT_API_KEY)
 
    comments = []
    next_page_token = None
 
    while len(comments) < max_comments:
        batch_size = min(100, max_comments - len(comments))  # API max per page is 100
 
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=batch_size,
            pageToken=next_page_token,
            textFormat="plainText",
            order="relevance",  # top comments first
        ).execute()
 
        for item in response.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(text)
 
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break  # no more pages
 
    return comments[:max_comments]


def analyse(comments: list[str]) -> list[dict]:
    """Run the sentiment model on each comment and return structured results."""
    results = []
    pos=0
    neg=0
    neu=0
    preprocessed_comments = [preprocess_comment(comment) for comment in comments]
    transformed_comments = vectorizer.transform(preprocessed_comments)
    predictions = model.predict(transformed_comments).tolist()  # Convert to list
    for predicts in predictions:
        if (predicts == 0):
            label = 'negative'
            neg+=1
        elif (predicts == 2):
            label = 'positive'
            pos+=1
        else:
            label = 'neutral'
            neu+=1

        results.append(label)
    
    answer = dict(zip(comments, results))
    return answer, pos, neg, neu

def print_results(answer: dict, pos:int, neg:int, neu:int):

    print("\n" + "═" * 60)
    print("  OVERALL SENTIMENT BREAKDOWN")
    print(f"positive: {pos}, negative: {neg}, neutral: {neu}")
    print("═" * 60)
    
    EMOJIS = {"positive": "😊", "negative": "😠", "neutral": "😐"}
    print("\n" + "═" * 60)
    print("  COMMENT-BY-COMMENT RESULTS")
    print("═" * 60)

    

    for c, r in answer.items():
        comment_preview = c.replace("\n", " ")
        if len(comment_preview) > 80:
            comment_preview = comment_preview[:77] + "..."

        print(f"{EMOJIS[r]}: {comment_preview}")



def main():
    print("\n🎬  YouTube Comment Sentiment Analyser")
    print("─" * 40)
 
    url = input("Enter YouTube video URL: ").strip()
    if not url:
        print("No URL entered. Exiting.")
        return
 
    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        print(f"\n❌  {e}")
        return
 
    print(f"\n⏳  Fetching up to 100 comments for video ID: {video_id} ...")
    try:
        comments = scrape_comments(video_id, max_comments=100)
    except Exception as e:
        print(f"\n❌  Failed to fetch comments: {e}")
        print("    Check that your YT_API_KEY is correct and the video has comments enabled.")
        return
 
    if not comments:
        print("\n⚠️  No comments found (video may have comments disabled).")
        return
 
    print(f"✅  Fetched {len(comments)} comments. Running sentiment analysis...\n")
 
    answer, pos, neg, neu = analyse(comments)
    print_results(answer, pos, neg, neu)
 
 
if __name__ == "__main__":
    main()
