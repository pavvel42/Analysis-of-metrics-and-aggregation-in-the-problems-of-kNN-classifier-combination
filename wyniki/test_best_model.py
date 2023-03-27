import pandas as pd
from scipy.stats import mannwhitneyu, f_oneway, ttest_ind, shapiro
import numpy as np

# Wczytanie danych z pliku xlsx do DataFrame
data = {}
sheets = ['colon', 'prostate', 'leukemia', 'ovarian', 'lymphoma']
metrics = ['canberra', 'manhattan', 'chebyshev']
aggregations = ['arithmetic', 'quadratic']
n_neighbors = 7

for sheet in sheets:
    df = pd.read_excel('roboczy.xlsx', sheet_name=sheet)
    filtered_df = df[(df['metric'].isin(metrics)) & (df['aggregation'].isin(aggregations)) & (df['n_neighbors'] == n_neighbors)]
    data[sheet] = (filtered_df['meanAUC'], filtered_df['mean_single_kNN_AUC'])

# Obliczenie średniej z odpowiednich kolumn i wykonywanie testów statystycznych
for sheet, (col_f, col_g) in data.items():
    try:
        col_f = [float(x) for x in col_f]
        col_g = [float(x) for x in col_g]

        col_g_sorted = sorted(col_g, reverse=True) #sortowanie
        col_g = col_g_sorted[::9] #co 9 wartość dla mean_single_kNN_AUC

        print(sheet.capitalize(), len(col_f), col_f)
        print(sheet.capitalize(), len(col_g), col_g)

        # # Wykonanie testu Shapiro-Wilka
        # shapiro_stat_f, shapiro_p_value_f = shapiro(col_f)
        # shapiro_stat_g, shapiro_p_value_g = shapiro(col_g)
        # print("Wynik testu Shapiro-Wilka dla {}: statystyka = {:.5f}, p-value = {:.8f}".format(sheet.capitalize(), shapiro_stat_f, shapiro_p_value_f))
        # print("Wynik testu Shapiro-Wilka dla {}: statystyka = {:.5f}, p-value = {:.8f}".format("Mean_single_kNN_AUC", shapiro_stat_g, shapiro_p_value_g))

        # Wykonanie f-testu
        f_stat, f_p_value = f_oneway(col_f, col_g)
        print("Wynik f-testu dla {}: statystyka = {:.5f}, p-value = {:.8f}".format(sheet.capitalize(), f_stat, f_p_value))

        # Wykonanie t-testu (dla małych próbek)
        t_stat, t_p_value = ttest_ind(col_f, col_g)
        print("Wynik t-testu dla {}: statystyka = {:.5f}, p-value = {:.8f}".format(sheet.capitalize(), t_stat, t_p_value))

        # Wykonanie testu Mann-Whitneya (dla dużych próbek)
        stat, p_value = mannwhitneyu(col_f, col_g, alternative='two-sided')
        print("Wynik testu Mann-Whitneya dla {}: statystyka = {:.5f}, p-value = {:.8f}".format(sheet.capitalize(), stat, p_value))

        # Obliczenie średniej, odchylenia standardowego i mediany dla kolumn
        mean_f = np.mean(col_f)
        std_f = np.std(col_f)
        median_f = np.median(col_f)

        mean_g = np.mean(col_g)
        std_g = np.std(col_g)
        median_g = np.median(col_g)

        # # Wypisanie wyników
        print("Średnia kolumny meanAUC dla {}: {:.5f}".format(sheet.capitalize(), mean_f))
        print("Średnia kolumny mean_single_kNN_AUC dla {}: {:.5f}".format(sheet.capitalize(), mean_g))
        print("Odchylenie standardowe kolumny meanAUC dla {}: {:.5f}".format(sheet.capitalize(), std_f))
        print("Odchylenie standardowe kolumny mean_single_kNN_AUC dla {}: {:.5f}".format(sheet.capitalize(), std_g))
        print("Mediana dla kolumny meanAUC dla {}: {:.5f}".format(sheet.capitalize(), median_f))
        print("Mediana dla kolumny mean_single_kNN_AUC dla {}: {:.5f}".format(sheet.capitalize(), median_g))
        
    except ValueError as e:
        print("Value Error occurred: {}".format(e))
        continue
    except ZeroDivisionError as e:
        print("ZeroDivisionError occurred: {}".format(e))
        continue
    except Exception as e:
        print("Unexpected Error occurred: {}".format(e))
        continue

