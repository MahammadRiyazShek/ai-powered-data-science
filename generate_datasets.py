import csv
import random
from pathlib import Path

random.seed(42)
root = Path('/home/user/industrial_ai_ds_project')

# HR dirty dataset
hr_rows = []
for i in range(1, 81):
    experience = random.randint(0, 25)
    rating = round(random.uniform(2.2, 5.0), 1)
    salary = 28000 + experience * 4200 + rating * 3200 + random.randint(-7000, 7000)
    salary = int(max(22000, salary))
    hr_rows.append([i, experience, rating, salary, random.choice(['Tech', 'Finance', 'HR', 'Operations'])])

# Inject dirtiness
hr_rows += [
    [81, '', 4.2, 91000, 'Tech'],
    [82, 12, '', 87000, 'Finance'],
    [83, 18, 4.8, '', 'Operations'],
    [84, 45, 4.0, 980000, 'Tech'],
    [85, -3, 3.1, 15000, 'HR'],
    [86, 9, 4.1, 87000, 'Finance'],
    [86, 9, 4.1, 87000, 'Finance'],
    [87, 21, 5.0, 155000, 'Operations'],
    [88, 2, 2.5, 31500, 'HR'],
    [89, 16, 4.4, 112000, 'Tech'],
]

with open(root / 'hr_dirty_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['EmployeeID', 'Experience', 'Rating', 'Salary', 'Department'])
    writer.writerows(hr_rows)

# Customer dataset with four natural clusters + a little noise
customer_rows = []
customer_id = 1
clusters = [
    {'age': (22, 34), 'spend': (1800, 3200), 'segment': 'Young Value'},
    {'age': (28, 42), 'spend': (4200, 7800), 'segment': 'Premium Regular'},
    {'age': (43, 60), 'spend': (2600, 5200), 'segment': 'Family Standard'},
    {'age': (20, 30), 'spend': (8000, 14000), 'segment': 'High Roller'},
]

for meta in clusters:
    for _ in range(30):
        age = random.randint(*meta['age'])
        spend = random.randint(*meta['spend'])
        visits = random.randint(1, 12)
        customer_rows.append([customer_id, age, spend, visits, meta['segment']])
        customer_id += 1

# Add noisy points / duplicates / missing values
customer_rows += [
    [121, '', 2500, 3, 'Unknown'],
    [122, 26, '', 5, 'Unknown'],
    [123, 19, 22000, 2, 'Outlier'],
    [124, 74, 600, 1, 'Outlier'],
    [125, 31, 7100, 9, 'Premium Regular'],
    [125, 31, 7100, 9, 'Premium Regular'],
]

with open(root / 'customer_segmentation_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['CustomerID', 'Age', 'MonthlySpend', 'VisitsPerMonth', 'LegacyLabel'])
    writer.writerows(customer_rows)

print('Datasets generated.')
