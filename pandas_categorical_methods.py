import pandas as pd
import numpy as np

# Create a sample dataset with categorical variables
data = {
    'customer_id': range(1, 11),
    'age': [25, 32, 45, 28, 35, 42, 30, 27, 38, 33],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    'product_category': ['Electronics', 'Clothing', 'Electronics', 'Food', 'Clothing',
                        'Electronics', 'Food', 'Clothing', 'Electronics', 'Food'],
    'purchase_amount': [1200, 350, 1800, 150, 450, 2200, 200, 380, 1600, 180],
    'satisfaction_level': ['High', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low']
}

# Create DataFrame
df = pd.DataFrame(data)

# 1. Converting columns to categorical type
df['gender'] = df['gender'].astype('category')
df['product_category'] = df['product_category'].astype('category')
df['satisfaction_level'] = df['satisfaction_level'].astype('category')

print("\n1. DataFrame with categorical columns:")
print(df.dtypes)

# 2. Value counts for categorical columns
print("\n2. Value counts for gender:")
print(df['gender'].value_counts())

print("\nValue counts for product category:")
print(df['product_category'].value_counts())

# 3. Cross-tabulation (crosstab)
print("\n3. Cross-tabulation of gender and product category:")
print(pd.crosstab(df['gender'], df['product_category']))

# 4. GroupBy operations with categorical data
print("\n4. Average purchase amount by product category:")
print(df.groupby('product_category')['purchase_amount'].mean())

# 5. Pivot table with categorical data
print("\n5. Pivot table - average purchase amount by gender and product category:")
print(pd.pivot_table(df, 
                    values='purchase_amount',
                    index='gender',
                    columns='product_category',
                    aggfunc='mean'))

# 6. Categorical methods
print("\n6. Categories in product_category:")
print(df['product_category'].cat.categories)

# 7. Adding a new category
df['product_category'] = df['product_category'].cat.add_categories(['Books'])
print("\n7. Categories after adding 'Books':")
print(df['product_category'].cat.categories)

# 8. Renaming categories
df['satisfaction_level'] = df['satisfaction_level'].cat.rename_categories({
    'Low': 'Poor',
    'Medium': 'Average',
    'High': 'Excellent'
})
print("\n8. Renamed satisfaction levels:")
print(df['satisfaction_level'].value_counts())

# 9. Sorting categories
df['satisfaction_level'] = df['satisfaction_level'].cat.reorder_categories(
    ['Poor', 'Average', 'Excellent'],
    ordered=True
)
print("\n9. Sorted satisfaction levels:")
print(df['satisfaction_level'].value_counts())

# 10. Filtering based on categorical values
print("\n10. Customers with high satisfaction:")
print(df[df['satisfaction_level'] == 'Excellent'])

# 11. One-hot encoding of categorical variables
print("\n11. One-hot encoding of product categories:")
print(pd.get_dummies(df['product_category']))

# 12. Combining categorical columns
print("\n12. Combined analysis of gender and satisfaction:")
print(df.groupby(['gender', 'satisfaction_level'])['purchase_amount'].mean()) 