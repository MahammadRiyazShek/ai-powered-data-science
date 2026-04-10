import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_FILE = 'customer_segmentation_data.csv'
NEW_CUSTOMER = {'Age': 25, 'MonthlySpend': 2500}
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


def clean_customer_data(df):
    original_rows = len(df)
    df = df.drop_duplicates().copy()

    for col in ['Age', 'MonthlySpend', 'VisitsPerMonth']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    df = df[
        (df['Age'].between(18, 75)) &
        (df['MonthlySpend'].between(500, 20000)) &
        (df['VisitsPerMonth'].between(1, 20))
    ].copy()

    df = remove_outliers_iqr(df, ['Age', 'MonthlySpend'])
    df = df.reset_index(drop=True)
    return df, {'original_rows': original_rows, 'final_rows': len(df)}


def choose_best_k(X_scaled):
    scores = {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def label_segment(age, spend):
    if spend >= 7000:
        return 'VIP / High Roller'
    if age <= 35 and spend < 3500:
        return 'Young Value'
    if 3500 <= spend < 7000:
        return 'Premium Regular'
    return 'Family Standard'


def main():
    raw_df = pd.read_csv(DATA_FILE)
    clean_df, summary = clean_customer_data(raw_df)

    X = clean_df[['Age', 'MonthlySpend']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, silhouette_map = choose_best_k(X_scaled)

    model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    labels = model.fit_predict(X_scaled)
    clean_df['Cluster'] = labels

    score = silhouette_score(X_scaled, labels)
    centers = scaler.inverse_transform(model.cluster_centers_)

    profiles = []
    for idx, (age, spend) in enumerate(centers):
        profiles.append({
            'cluster': idx,
            'avg_age': round(float(age), 2),
            'avg_spend': round(float(spend), 2),
            'label': label_segment(age, spend)
        })

    new_scaled = scaler.transform(pd.DataFrame([NEW_CUSTOMER]))
    new_cluster = model.predict(new_scaled)[0]
    new_label = next(item['label'] for item in profiles if item['cluster'] == new_cluster)

    print('\nCUSTOMER SEGMENTATION AUDIT')
    print('-' * 60)
    print(f"Original rows: {summary['original_rows']}")
    print(f"Final cleaned rows: {summary['final_rows']}")
    print(f"Selected K: {best_k}")
    print(f"Silhouette score: {score:.4f}")
    print('\nSilhouette by K:')
    for k, val in silhouette_map.items():
        print(f"  K={k}: {val:.4f}")

    print('\nCluster profiles:')
    for item in sorted(profiles, key=lambda x: x['cluster']):
        print(f"Cluster {item['cluster']}: Age={item['avg_age']}, Spend=${item['avg_spend']:,.2f}, Label={item['label']}")

    print(
        f"\nNew customer classification -> Age {NEW_CUSTOMER['Age']}, "
        f"Spend ${NEW_CUSTOMER['MonthlySpend']:,.2f}: Cluster {new_cluster} ({new_label})"
    )


if __name__ == '__main__':
    main()
