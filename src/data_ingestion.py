import pandas as pd

def import_data():
    sentiment_df = pd.read_csv("data/raw/Sentiment_Mental_health_dataset.csv")
    return sentiment_df