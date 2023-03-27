import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np

# Wczytanie danych z pliku xlsx do DataFrame
data = {}
# sheets = ['ovarian', 'leukemia', 'colon', 'prostate', 'lymphoma']
sheets = ['lymphoma']
for sheet in sheets:
    df = pd.read_excel('roboczy.xlsx', sheet_name=sheet)
    data[sheet] = (df['meanAUC'], df['mean_single_kNN_AUC'])

# Obliczenie średniej z odpowiednich kolumn i wykonywanie testu Mann-Whitneya
for sheet, (col_f, col_g) in data.items():
    col_f = [float(x) for x in col_f]
    col_g = [float(x) for x in col_g]

    # col_g = col_g[:-2] #usunięcie mediany i średniej z excela
    col_g_sorted = sorted(col_g, reverse=True) #sortowanie
    col_g = col_g_sorted[::16] #co 16 wartość dla mean_single_kNN_AUC

    col_g_mean = np.mean(col_g)
    col_f_mean = np.mean(col_f)
    col_f_std = np.std(col_f)
    col_g_std = np.std(col_g)
    stat, p_value = mannwhitneyu(col_f, col_g, alternative='two-sided')
    # print(sheet, col_g)
    print("Wynik testu Mann-Whitneya dla {}: statystyka = {:.5f}, p-value = {:.8f}".format(sheet.capitalize(), stat, p_value))
    print("Średnia kolumny meanAUC dla {}: {:.5f}".format(sheet.capitalize(), col_f_mean))
    print("Średnia kolumny mean_single_kNN_AUC dla {}: {:.5f}".format(sheet.capitalize(), col_g_mean))
    print("Odchylenie standardowe kolumny meanAUC dla {}: {:.5f}".format(sheet.capitalize(), col_f_std))
    print("Odchylenie standardowe kolumny mean_single_kNN_AUC dla {}: {:.5f}".format(sheet.capitalize(), col_g_std))

