import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = 'hr_dirty_dataset.csv'
PREDICT_EXPERIENCE = 15
RANDOM_STATE = 42


def remove_outliers_iqr(df, columns):
    result = df.copy()
    for col in columns:
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        result = result[(result[col] >= lower) & (result[col] <= upper)]
    return result


def clean_hr_data(df):
    original_rows = len(df)
    df = df.drop_duplicates().copy()
    after_duplicates = len(df)

    for col in ['Experience', 'Rating', 'Salary']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Remove unrealistic ranges
    df = df[
        (df['Experience'].between(0, 40)) &
        (df['Rating'].between(1, 5)) &
        (df['Salary'].between(20000, 250000))
    ].copy()

    # Remove statistical outliers
    df = remove_outliers_iqr(df, ['Experience', 'Salary'])
    df = df.sort_values('Experience').reset_index(drop=True)

    summary = {
        'original_rows': original_rows,
        'after_duplicates': after_duplicates,
        'final_rows': len(df),
    }
    return df, summary


def main():
    raw_df = pd.read_csv(DATA_FILE)
    clean_df, summary = clean_hr_data(raw_df)

    X = clean_df[['Experience']]
    y = clean_df['Salary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    predicted_salary = model.predict(pd.DataFrame({'Experience': [PREDICT_EXPERIENCE]}))[0]

    print('\nSALARY PREDICTION AUDIT')
    print('-' * 60)
    print(f"Original rows: {summary['original_rows']}")
    print(f"Rows after duplicate removal: {summary['after_duplicates']}")
    print(f"Rows after full cleaning: {summary['final_rows']}")
    print(f"Model coefficient (slope): {model.coef_[0]:.2f}")
    print(f"Model intercept: {model.intercept_:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Predicted salary for {PREDICT_EXPERIENCE} years of experience: ${predicted_salary:,.2f}")
    print('\nTop cleaned rows:')
    print(clean_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
