import os
import pandas as pd
from dotenv import load_dotenv

# ============================================================================
# HUGGING FACE AUTHENTICATION & DATA SETUP
# ============================================================================

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

print("Downloading and cleaning laws segment from OmniCorpus...")

path = "hf://datasets/mongramosjr/philippine-omnicorpus/data/philippine_laws.parquet"
# Load Philippine legal documents from Hugging Face OmniCorpus dataset
df = pd.read_parquet(
    path, 
    storage_options={"token": hf_token}
)

CAT_COL = 'label' 

# ============================================================================
# DATA CLEANING & PREPROCESSING
# ============================================================================

print(f"Original dataset size: {len(df)} rows")

# Remove documents with missing text or category labels
df = df.dropna(subset=['text', CAT_COL])

# Filter out short documents (less than 50 tokens) to ensure sufficient content
df = df[df['text'].str.split().str.len() > 50]

print(f"Size after removing empty/short rows: {len(df)} rows")

# Define valid document categories for Philippine legal corpus
valid_statutes = ['Republic Acts', 'Commonwealth Act', 'Acts']
valid_decisions = ['Decisions / Signed Resolutions', 'Decisions / Sign Resolutions']
valid_regs = ['Executive Orders', 'Memorandum Circulars', 'Letter of Instruction']

# ============================================================================
# STRATIFIED SAMPLING BY DOCUMENT TYPE
# ============================================================================

# Partition documents by legal category for stratified sampling
statutes_df = df[df[CAT_COL].isin(valid_statutes)]
decisions_df = df[df[CAT_COL].isin(valid_decisions)]
regs_df = df[df[CAT_COL].isin(valid_regs)]

try:
    # Sample from each category with fixed seed for reproducibility (total: 2,400 docs)
    statutes = statutes_df.sample(n=900, random_state=42)
    decisions = decisions_df.sample(n=1200, random_state=42)
    regs = regs_df.sample(n=300, random_state=42)

    # Combine sampled documents into unified corpus
    plrb_corpus = pd.concat([statutes, decisions, regs])
    
    # ================================================================
    # EXPORT & SUMMARY
    # ================================================================
    
    os.makedirs("data/raw", exist_ok=True)
    
    # Save cleaned and stratified corpus for downstream processing
    plrb_corpus.to_csv("data/raw/master_legal_corpus.csv", index=False)
    
    print("\n--- Cleaning and Extraction Complete ---")
    print(f"Successfully saved 2,400 strictly validated documents to data/raw/master_legal_corpus.csv")
    print("Document Breakdown:")
    print(plrb_corpus[CAT_COL].value_counts())

except ValueError as e:
    print(f"\nError during sampling: {e}")
    print(f"Available Statutes: {len(statutes_df)}")
    print(f"Available Decisions: {len(decisions_df)}")
    print(f"Available Regulations: {len(regs_df)}")