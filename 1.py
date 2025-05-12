import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "pemilu since:2024-01-01 until:2024-01-31 lang:id"
tweets = []
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i >= 200:  # ambil 200 tweet
        break
    tweets.append([tweet.date, tweet.user.username, tweet.content])

df = pd.DataFrame(tweets, columns=["Tanggal", "Username", "Isi_Tweet"])
df.to_csv("tweet_pemilu.csv", index=False)
print("Selesai scraping!")
