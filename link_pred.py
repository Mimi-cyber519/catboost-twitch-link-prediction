import time
start_time = time.time()
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import json
import random

df=pd.read_csv('edges.csv')
df_profiles=pd.read_csv('target.csv').set_index('new_id')
with open('features.json', 'r') as f:
    features_json = json.load(f)

random.seed(42)
G=nx.from_pandas_edgelist(df,'from','to')
communities_list=nx.community.louvain_communities(G, seed=42)
com = {}
for com_id, community in enumerate(communities_list):
    for user_id in community:
        com[user_id] = com_id

df_all = pd.DataFrame({
    'connections': dict(G.degree()),
    'rate': nx.pagerank(G),
    'community': com
}).join(df_profiles[['days', 'mature', 'views', 'partner']])

X = []
y = []

all_users = list(df_all.index)

def get_pair_features(u1, u2):
    common_friends = len(nx.common_neighbors(G, u1, u2))
    same_comm = 1 if df_all.loc[u1, 'community'] == df_all.loc[u2, 'community'] else 0
    views_diff = abs(df_all.loc[u1, 'views'] - df_all.loc[u2, 'views'])
    try:
        tags_1 = set(features_json[str(u1)])
        tags_2 = set(features_json[str(u2)])
        common_interests = len(tags_1 & tags_2)
    except KeyError:
        common_interests = 0

    return [common_friends, same_comm, views_diff, common_interests]

for index, row in df.iterrows():
    u1 = row['from']
    u2 = row['to']
    features = get_pair_features(u1, u2)
    X.append(features)
    y.append(1)

positive_count = len(df)
while len(X) < positive_count * 2:
    u1 = random.choice(all_users)
    u2 = random.choice(all_users)

    if u1 != u2 and not G.has_edge(u1, u2):
        features = get_pair_features(u1, u2)
        X.append(features)
        y.append(0)

X_df = pd.DataFrame(X, columns=['common_friends', 'same_comm', 'views_diff', 'common_interests'])
y_series = pd.Series(y)

train_start = time.time()
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
)

print(f"Обучение: {X_train.shape[0]} пар, Валидация: {X_test.shape[0]} пар.")


model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)


y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"ROC-AUC на валидации: {auc_score:.4f}")
print("\nОтчет о классификации (Precision / Recall):")
print(classification_report(y_test, model.predict(X_test)))

total_time = time.time() - start_time
print(f"\n Общее время выполнения: {total_time:.2f} секунд")
train_time = time.time() - train_start
print(f" Время обучения модели: {train_time:.2f} сек")

importance = model.get_feature_importance()
feature_imp_df = pd.DataFrame({
    'Признак': X_df.columns,
    'Важность (%)': importance
}).sort_values(by='Важность (%)', ascending=True)

plt.figure(figsize=(10, 5))
plt.barh(feature_imp_df['Признак'], feature_imp_df['Важность (%)'], color='royalblue')
plt.xlabel('Важность признака в %')
plt.title('Какие факторы сильнее всего влияют на предсказание связи?')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
