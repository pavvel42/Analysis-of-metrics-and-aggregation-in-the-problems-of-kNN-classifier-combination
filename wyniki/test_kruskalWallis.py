import pandas as pd
from scipy.stats import kruskal
import sys

# Wczytaj dane z pliku
df = pd.read_excel('roboczy.xlsx', sheet_name='wykres1')

# Listy z ilością sąsiadów i ilością zgięć
neighbors = [3, 5, 7]
folds = [2, 5, 10, 20]

# Lista z metrykami
metrics = ['braycurtis', 'canberra', 'euclidean', 'manhattan', 'minkowski'] #bez 'chebyshev'

# sys.stdout = open("test_kruskalWallis_Wyniki.txt", "w")
# Pętla po ilości sąsiadów i ilości zgięć
for n in neighbors:
    for f in folds:
        # Utwórz listę z wynikami dla danej ilości sąsiadów i ilości zgięć
        results = []
        for metric in metrics:
            if metric != 'minkowski':
                data = df[(df['n_neighbors'] == n) & (df['s'] == f) & (df['metric'] == metric)]['meanAUC'].values
                results.append(data)
                # print(f"neighbors = {n}, folds = {f}, metric = {metric}: {data}")
            else:
                data = df[(df['n_neighbors'] == n) & (df['s'] == f) & (df['p'] == 1.5) & (df['metric'] == metric)]['meanAUC'].values
                results.append(data)
                # print(f"neighbors = {n}, folds = {f}, metric = {metric}: {data}")
            
        # Przeprowadź test Kruskala-Wallisa
        stat, p_value = kruskal(*results)
        # stat, p_value = kruskal(results[0],results[1])
        print(f"neighbors = {n}, folds = {f}: stat = {stat}, p_value = {p_value}")

# sys.stdout.close()