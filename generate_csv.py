# generate_csv.py (extended dataset)
import pandas as pd

texts = [
    "The government announced a new education policy.",
    "NASA found evidence of ancient life on Mars.",
    "Claim your free iPhone now by clicking this link!",
    "Scientists invented a pill that reverses aging.",
    "This one weird trick will make you rich overnight!",
    "The Prime Minister addressed the nation today.",
    "Aliens are secretly living among us, report says.",
    "The stock market hits an all-time high.",
    "Win a vacation now! Limited time offer!",
    "New study shows eating chocolate boosts brain power.",
    "Vaccines cause autism, according to fake news site.",
    "Electric cars are the future, says Elon Musk.",
    "COVID-19 vaccine implants microchips - hoax debunked.",
    "UN announces mission to eliminate world hunger.",
    "Your computer is infected! Download this antivirus.",
    "Global temperatures are rising at an alarming rate.",
    "Earn $1000/day from home with no skills!",
    "India successfully lands spacecraft on the moon.",
    "World’s largest diamond found in Africa.",
    "Click to watch the shocking video now!"
]

labels = [
    "REAL", "REAL", "FAKE", "REAL", "FAKE", "REAL", "FAKE", "REAL", "FAKE", "REAL",
    "FAKE", "REAL", "FAKE", "REAL", "FAKE", "REAL", "FAKE", "REAL", "REAL", "FAKE"
]

df = pd.DataFrame({'text': texts, 'label': labels})
df.to_csv("fake_or_real_news.csv", index=False)
print("✅ Larger CSV created: fake_or_real_news.csv")
