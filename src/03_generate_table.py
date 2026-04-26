import pandas as pd

print("Generating Table 1 Statistics from processed data...")

# 1. Load your processed dataset
df = pd.read_csv("data/processed/module1_processed_passages.csv")

# 2. Map your specific labels to the paper's broad categories
category_mapping = {
    'Republic Acts': 'Republic Acts / Pres. Decrees',
    'Acts': 'Republic Acts / Pres. Decrees',
    'Commonwealth Act': 'Republic Acts / Pres. Decrees',
    'Decisions / Signed Resolutions': 'Supreme Court Decisions',
    'Executive Orders': 'Administrative Regulations',
    'Memorandum Circulars': 'Administrative Regulations',
    'Letter of Instruction': 'Administrative Regulations'
}
df['Broad_Category'] = df['document_type'].map(category_mapping)

# 3. Calculate the number of tokens per passage
df['Tokens'] = df['passage_text'].astype(str).apply(lambda x: len(x.split()))

# 4. Create a pivot table to count languages per category
lang_counts = pd.crosstab(df['Broad_Category'], df['language'])

# Ensure required columns exist
for col in ['English', 'Filipino', 'Code-Switched']:
    if col not in lang_counts.columns:
        lang_counts[col] = 0

# 5. Get total passages and average tokens per category
agg_df = df.groupby('Broad_Category').agg(
    Total=('passage_text', 'count'),
    Avg_Tokens=('Tokens', 'mean')
)

# 6. Merge the data together
table = agg_df.join(lang_counts)

# Keep only the columns requested in the image (Dropping the 'Other' category)
table = table[['Total', 'English', 'Filipino', 'Code-Switched', 'Avg_Tokens']]

# Reorder rows to match your image exactly
row_order = ['Republic Acts / Pres. Decrees', 'Supreme Court Decisions', 'Administrative Regulations']
table = table.reindex(row_order)

# 7. Calculate the "Total" bottom row
total_row = pd.DataFrame({
    'Total': [table['Total'].sum()],
    'English': [table['English'].sum()],
    'Filipino': [table['Filipino'].sum()],
    'Code-Switched': [table['Code-Switched'].sum()],
    'Avg_Tokens': [df['Tokens'].mean()]
}, index=['Total'])

# Append total row
final_table = pd.concat([table, total_row])

# Round average tokens to whole numbers
final_table['Avg_Tokens'] = final_table['Avg_Tokens'].round().astype(int)

# 8. Save to CSV and Print
output_path = "data/processed/Table_1_Corpus_Statistics.csv"
final_table.to_csv(output_path, index_label="Document Type")

print("\n" + "="*80)
print("   TABLE 1: PLRB CORPUS STATISTICS (ACTUAL EMPIRICAL DATA)")
print("="*80)
print(final_table.to_string())
print("="*80)
print(f"\nDone! You can open '{output_path}' in Excel and copy it directly into your Word document.")