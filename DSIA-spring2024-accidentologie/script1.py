import pandas as pd
import numpy as np

base_path_train = 'TRAIN/BAAC-Annee-2019/'
#data_test = pd.read_csv('chemin_vers_le_fichier_test.csv')

caracteristiques = pd.read_csv(base_path_train + 'caracteristiques_2019_.csv', sep=';')
lieux = pd.read_csv(base_path_train + 'lieux_2019_.csv', sep=';')
usagers = pd.read_csv(base_path_train + 'usagers_2019_.csv', sep=';')
vehicules = pd.read_csv(base_path_train + 'vehicules_2019_.csv', sep=';')

#drop unnamed colonnes
caracteristiques.drop(columns=['Unnamed: 0'], inplace=True)
lieux.drop(columns=['Unnamed: 0'], inplace=True)
usagers.drop(columns=['Unnamed: 0'], inplace=True)
vehicules.drop(columns=['Unnamed: 0'], inplace=True)


# Remplissage des valeurs manquantes dans 'caracteristiques'
caracteristiques['adr'].fillna('Inconnue', inplace=True)

# Remplissage des valeurs manquantes dans 'lieux'
lieux['voie'].fillna('Non spécifiée', inplace=True)
lieux['v1'].fillna('Non spécifié', inplace=True)
lieux['v2'].fillna('Non spécifié', inplace=True)
lieux['lartpc'].fillna(0, inplace=True)  # ou laisser comme NaN si non utilisé
lieux['larrout'].fillna(0, inplace=True)  # ou laisser comme NaN si non utilisé

# Remplissage des valeurs manquantes dans 'vehicules'
vehicules['occutc'].fillna(0, inplace=True)  # Rempli avec 0 si pas un transport en commun

# Vérification après nettoyage
print(caracteristiques.isnull().sum())
print(lieux.isnull().sum())
print(vehicules.isnull().sum())

