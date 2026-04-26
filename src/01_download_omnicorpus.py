import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

print("Downloading specific laws segment from OmniCorpus...")

path = "hf://datasets/mongramosjr/philippine-omnicorpus/data/philippine_laws.parquet"

df = pd.read_parquet(
    path, 
    storage_options={"token": hf_token}
)

print("\n--- Diagnostic: Column Names Found ---")
print(df.columns.tolist())

CAT_COL = 'label' 

try:
    statutes = df[df[CAT_COL].str.contains('Republic Act|Commonwealth Act|Act', case=False, na=False)].sample(n=900, random_state=42)
    
    decisions = df[df[CAT_COL].str.contains('Decision', case=False, na=False)].sample(n=1200, random_state=42)
    
    regs = df[df[CAT_COL].str.contains('Executive Order|Memorandum Circular|Instruction', case=False, na=False)].sample(n=300, random_state=42)

    plrb_corpus = pd.concat([statutes, decisions, regs])
    
    os.makedirs("data/raw", exist_ok=True)
    
    plrb_corpus.to_csv("data/raw/master_legal_corpus.csv", index=False)
    print("\nStep 1 Complete: 2,400 documents saved to data/raw/master_legal_corpus.csv")

except ValueError as e:
    print(f"\nError: Could not find enough documents in one of the categories. {e}")