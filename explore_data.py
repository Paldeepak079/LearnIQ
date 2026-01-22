import pandas as pd

# Load dataset
df = pd.read_csv('student_dataset.csv')

print('='*80)
print('DATASET OVERVIEW')
print('='*80)
print(f'\nShape: {df.shape[0]} students x {df.shape[1]} features')

print('\n' + '='*80)
print('ALL COLUMNS')
print('='*80)
for i, col in enumerate(df.columns, 1):
    print(f'{i:2d}. {col}')

print('\n' + '='*80)
print('DATA TYPES')
print('='*80)
print(df.dtypes)

print('\n' + '='*80)
print('MISSING VALUES')
print('='*80)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing': missing.values,
    'Percentage': missing_pct.values
})
print(missing_df[missing_df['Missing'] > 0])

print('\n' + '='*80)
print('SAMPLE DATA (First 3 rows)')
print('='*80)
print(df.head(3))

print('\n' + '='*80)
print('NUMERIC COLUMNS STATISTICS')
print('='*80)
print(df.describe())
