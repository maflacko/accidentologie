import pandas as pd

def clean_caracteristiques(filepath, delimiter=','):
    df = pd.read_csv(filepath, delimiter=delimiter)
    
    # Suppression d'une colonne non désirée si elle existe
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)
    
    df['hrmn'] = pd.to_datetime(df['hrmn'], format='%H%M', errors='coerce').dt.time
    df['adr'].fillna('Inconnue', inplace=True)
    df['atm'].fillna(-1, inplace=True)
    df.drop(columns='gps', inplace=True, errors='ignore')  # Ignore errors if 'gps' does not exist

    # Mode et imputation pour 'hrmn'
    hrmn_mode = df['hrmn'].mode()
    if not hrmn_mode.empty:
        df['hrmn'].fillna(hrmn_mode[0], inplace=True)
    else:
        df['hrmn'].fillna(pd.Timestamp('00:00').time(), inplace=True)

    df['lat'] = df['lat'].astype(str).str.replace(',', '.').astype(float)
    df['long'] = df['long'].astype(str).str.replace(',', '.').astype(float)
    df['lat'].fillna(0.0, inplace=True)
    df['long'].fillna(0.0, inplace=True)

    return df


def clean_lieux(filepath, delimiter=','):
    df = pd.read_csv(filepath,  delimiter=delimiter)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)
    
    for col in ['lartpc', 'larrout']:
        df['lartpc'] = df['lartpc'].astype(str).str.replace(',', '.').astype(float)
        df['larrout'] = df['larrout'].astype(str).str.replace(',', '.').astype(float)
        df[col].fillna(0, inplace=True)
    for col in ['circ', 'vosp', 'prof', 'plan', 'surf', 'infra', 'situ']:
        df[col].fillna(-1, inplace=True)
    vma_mode = df['vma'].mode()[0]
    df['vma'].fillna(vma_mode, inplace=True)
    nbv_median = df['nbv'].median()
    df['nbv'].fillna(nbv_median, inplace=True)
    df.drop(columns=['v1', 'v2', 'voie', 'pr', 'pr1'], inplace=True)
    if 'env1' in df.columns:
        df.drop(columns='env1', inplace=True)
    return df

def clean_vehicules(filepath, delimiter=','):
    df = pd.read_csv(filepath,  delimiter=delimiter)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)
    
    df['occutc'].fillna(0, inplace=True)
    df['motor'].fillna(-1, inplace=True)
    for col in ['senc', 'obs', 'obsm', 'choc', 'manv']:
        df[col].fillna(-1, inplace=True)
    return df

def clean_usagers(filepath, delimiter=','):
    df = pd.read_csv(filepath,  delimiter=delimiter)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)
    if 'secu' in df.columns:
        df.drop(columns='secu', inplace=True)
    
    df['actp'] = pd.to_numeric(df['actp'], errors='coerce')
    df['actp'].fillna(-1, inplace=True)
    df['secu1'].fillna(-1, inplace=True)
    df['secu2'].fillna(-1, inplace=True)
    df['secu3'].fillna(-1, inplace=True)
    an_nais_median = df['an_nais'].median()
    df['an_nais'].fillna(an_nais_median, inplace=True)
    for col in ['place', 'trajet', 'locp', 'etatp']:
        df[col].fillna(-1, inplace=True)
    return df


base_path_test = 'TEST/TEST/'
base_path_train = 'TRAIN/BAAC-Annee-2019/'

# Exemple d'utilisation pour les données de test
caracteristiques_test = clean_caracteristiques(base_path_test + 'CARACTERISTIQUES.csv')
#print("Info des caractéristiques test nettoyées:")
#caracteristiques_test.info()
#print("\nValeurs manquantes test dans chaque colonne:")
#print(caracteristiques_test.isnull().sum())
caracteristiques_train = clean_caracteristiques(base_path_train + 'caracteristiques_2019_.csv', delimiter=';')
#print("Info des caractéristiques train nettoyées:")
#caracteristiques_train.info()
#print("\nValeurs manquantes train dans chaque colonne:")
#print(caracteristiques_train.isnull().sum())

lieux_train = clean_lieux(base_path_train + 'lieux_2019_.csv', delimiter=';')
lieux_test = clean_lieux(base_path_test + 'LIEUX.csv')

print("Info des lieux test nettoyées:")
lieux_test.info()
print("\nValeurs manquantes test dans chaque colonne:")
print(lieux_test.isnull().sum())
print("Info des lieux train nettoyées:")
lieux_train.info()
print("\nValeurs manquantes train dans chaque colonne:")
print(lieux_train.isnull().sum())

vehicules_train = clean_vehicules(base_path_train + 'vehicules_2019_.csv',  delimiter=';')
vehicules_test = clean_vehicules(base_path_test + 'VEHICULES.csv')
print("Info des vehicules test nettoyées:")
vehicules_test.info()
print("\nValeurs manquantes test dans chaque colonne:")
print(vehicules_test.isnull().sum())
print("Info des lieux train nettoyées:")
vehicules_train.info()
print("\nValeurs manquantes train dans chaque colonne:")
print(vehicules_train.isnull().sum())

# Exemple d'utilisation pour les données d'entraînement
usagers_train = clean_usagers(base_path_train + 'usagers_2019_.csv',  delimiter=';')
usagers_test = clean_usagers(base_path_test + 'USAGERS.csv')



#print(vehicules_train.head())
#print(vehicules_test.head())


print(usagers_test.head())
print("Info des usagers test nettoyées:")
usagers_test.info()
print("\nValeurs manquantes test dans chaque colonne:")
print(usagers_test.isnull().sum())
print("Info des lieux train nettoyées:")
usagers_train.info()
print("\nValeurs manquantes train dans chaque colonne:")
print(usagers_train.isnull().sum())

print(usagers_train.head())
print(usagers_test.head())


#ajouter la prediction grave