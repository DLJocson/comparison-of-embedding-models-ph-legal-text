import pandas as pd
import unicodedata
import re
import os
from langdetect import detect_langs

def normalize_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKC', text)
    # Collapse multiple whitespaces/newlines into a single space
    return re.sub(r'\s+', ' ', text).strip()

def segment_passages(text, window=300, stride=50):
    tokens = text.split()
    return [" ".join(tokens[i:i + window]) for i in range(0, len(tokens), window - stride)]

def identify_language(text):
    try:
        predictions = detect_langs(text)
        res = {l.lang: l.prob for l in predictions}
        
        # Code-switching heuristic: If both Tagalog (tl) and English (en) are > 20%
        if 'en' in res and 'tl' in res and min(res['en'], res['tl']) > 0.20:
            return "Code-Switched"
        
        dominant = predictions[0].lang
        return "Filipino" if dominant == 'tl' else "English" if dominant == 'en' else "Other"
    except:
        # Handles edge cases where a chunk is only numbers/symbols
        return "Unknown"

print("Loading perfectly cleaned master corpus for preprocessing...")
df = pd.read_csv("data/raw/master_legal_corpus.csv")
final_data = []

print(f"Normalizing, chunking, and language-tagging {len(df)} documents. This will take a few minutes...")

for index, doc in df.iterrows():
    clean_text = normalize_text(doc['text'])
    passages = segment_passages(clean_text)
    
    for chunk in passages:
        final_data.append({
            "source_url": doc.get('url', 'Unknown'),
            "document_type": doc.get('label', 'Unknown'),
            "passage_text": chunk,
            "language": identify_language(chunk)
        })

# Create the processed folder if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

# Save the ML-ready dataset
processed_df = pd.DataFrame(final_data)
processed_df.to_csv("data/processed/module1_processed_passages.csv", index=False)

print("\n========================================")
print("   PROGRESS CHECK 1: FINAL DATA SUMMARY")
print("========================================")
print(f"Total Original Documents : {len(df)}")
print(f"Total Passages Generated : {len(processed_df)}")
print("\n--- Language Distribution ---")
print(processed_df['language'].value_counts())
print("========================================")
print("\nStep 2 Complete: Processed passages saved to data/processed/module1_processed_passages.csv")