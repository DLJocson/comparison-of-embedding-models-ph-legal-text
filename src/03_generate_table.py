import pandas as pd

# ============================================================================
# CORPUS STATISTICS TABLE GENERATION
# ============================================================================

print("Generating Table 1 Statistics from processed data...")

# Load processed passages with language tags and document metadata
df = pd.read_csv("data/processed/module1_processed_passages.csv")

# ============================================================================
# CATEGORY STANDARDIZATION
# ============================================================================

# Consolidate detailed document types into broad categories for corpus summary
# This aligns source document types with research framework categories
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

# ============================================================================
# TOKEN COUNT CALCULATION
# ============================================================================

# Compute token count per passage for corpus size metrics
df['Tokens'] = df['passage_text'].astype(str).apply(lambda x: len(x.split()))

# ============================================================================
# LANGUAGE DISTRIBUTION ANALYSIS
# ============================================================================

# Cross-tabulate languages detected across document categories
lang_counts = pd.crosstab(df['Broad_Category'], df['language'])

# Ensure all language columns exist with zero values if absent
for col in ['English', 'Filipino', 'Code-Switched']:
    if col not in lang_counts.columns:
        lang_counts[col] = 0

# ============================================================================
# CORPUS AGGREGATION BY CATEGORY
# ============================================================================

# Compute passage count and token statistics per document category
agg_df = df.groupby('Broad_Category').agg(
    Total=('passage_text', 'count'),
    Avg_Tokens=('Tokens', 'mean')
)

# Merge aggregation and language distribution tables
table = agg_df.join(lang_counts)

# ============================================================================
# TABLE FORMATTING & FINALIZATION
# ============================================================================

# Select columns in output order: total passages, language breakdown, avg tokens
table = table[['Total', 'English', 'Filipino', 'Code-Switched', 'Avg_Tokens']]

# Reorder rows by legal document category for publication format
row_order = ['Republic Acts / Pres. Decrees', 'Supreme Court Decisions', 'Administrative Regulations']
table = table.reindex(row_order)

# Calculate corpus-wide summary statistics
total_row = pd.DataFrame({
    'Total': [table['Total'].sum()],
    'English': [table['English'].sum()],
    'Filipino': [table['Filipino'].sum()],
    'Code-Switched': [table['Code-Switched'].sum()],
    'Avg_Tokens': [df['Tokens'].mean()]
}, index=['Total'])

# Append total row to corpus statistics table
final_table = pd.concat([table, total_row])

# Round average tokens to integers for readability
final_table['Avg_Tokens'] = final_table['Avg_Tokens'].round().astype(int)

# ============================================================================
# OUTPUT & SUMMARY
# ============================================================================

# Export table to CSV for integration into manuscript
output_path = "data/processed/Table_1_Corpus_Statistics.csv"
final_table.to_csv(output_path, index_label="Document Type")

# Display final corpus statistics summary
print("\n" + "="*80)
print("   TABLE 1: PLRB CORPUS STATISTICS (ACTUAL EMPIRICAL DATA)")
print("="*80)
print(final_table.to_string())
print("="*80)
print(f"\nDone! You can open '{output_path}' in Excel and copy it directly into your Word document.")