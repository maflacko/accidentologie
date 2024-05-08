import pandas as pd
###TEST
base_path_test = 'TEST/TEST/'

caracteristiques_test = pd.read_csv(base_path_test + 'CARACTERISTIQUES.csv')
lieux_test = pd.read_csv(base_path_test + 'LIEUX.csv')
usagers_test = pd.read_csv(base_path_test + 'USAGERS.csv')
vehicules_test = pd.read_csv(base_path_test + 'VEHICULES.csv')

##CARACTERISTIQUES
# Conversion des types de données
# Convertir 'hrmn' en format horaire correct s'il est en forme de chaîne de caractères numériques (e.g., 'HHMM')
caracteristiques_test['hrmn'] = pd.to_datetime(caracteristiques_test['hrmn'], format='%H%M', errors='coerce').dt.time
# Convertir 'lat' et 'long' en float
caracteristiques_test['lat'] = pd.to_numeric(caracteristiques_test['lat'], errors='coerce')
caracteristiques_test['long'] = pd.to_numeric(caracteristiques_test['long'], errors='coerce')
# Normaliser 'com' et 'dep' comme chaînes
caracteristiques_test['com'] = caracteristiques_test['com'].astype(str)
caracteristiques_test['dep'] = caracteristiques_test['dep'].astype(str)
caracteristiques_test['adr'].fillna('Inconnue', inplace=True)  # Imputer 'adr' avec 'Inconnue'
# Gestion des valeurs manquantes
# Imputer les valeurs manquantes de 'atm' par la mode 
caracteristiques_test['atm'].fillna(-1, inplace=True)
# Mise à jour des traitements selon les instructions
# Remplacer les valeurs manquantes de 'atm' par -1
caracteristiques_test['atm'].fillna(-1, inplace=True)
# Suppression de la colonne 'gps'
caracteristiques_test.drop(columns='gps', inplace=True)
# Gestion des valeurs manquantes de 'hrmn'
# Remplir les valeurs manquantes de 'hrmn' par la valeur la plus fréquente
hrmn_mode = caracteristiques_test['hrmn'].mode()[0]
caracteristiques_test['hrmn'].fillna(hrmn_mode, inplace=True)
# Conversion en float pour 'lat' et 'long' (déjà effectué, juste imputation des manquants)
lat_mean = caracteristiques_test['lat'].mean()
long_mean = caracteristiques_test['long'].mean()
caracteristiques_test['lat'].fillna(lat_mean, inplace=True)
caracteristiques_test['long'].fillna(long_mean, inplace=True)
# Vérification finale du dataframe nettoyé pour 'CARACTERISTIQUES.csv'
final_info_caracteristiques_test = {
    "Info finale": caracteristiques_test.info(),
    "Valeurs manquantes finales": caracteristiques_test.isnull().sum()
}

#print(final_info_caracteristiques_test["Valeurs manquantes finales"])


###LIEUX
# Nettoyage des colonnes avec formats mixtes en convertissant en numérique
for col in ['voie', 'pr', 'pr1', 'lartpc', 'larrout']:
        # Remplacer les virgules par des points puis convertir en numérique
        lieux_test[col] = lieux_test[col].astype(str).str.replace(',', '.').astype(float)
        lieux_test[col].fillna(0, inplace=True)

# Imputation des valeurs manquantes pour les colonnes avec un faible nombre de données manquantes
for col in ['circ', 'vosp', 'prof', 'plan', 'surf', 'infra', 'situ']:
    lieux_test[col].fillna(-1, inplace=True)

# Imputation pour 'vma' avec la mode
vma_mode = lieux_test['vma'].mode()[0]
lieux_test['vma'].fillna(vma_mode, inplace=True)
# Imputation pour 'nbv' avec la médiane
nbv_median = lieux_test['nbv'].median()
lieux_test['nbv'].fillna(nbv_median, inplace=True)

# Suppression des colonnes avec un très grand nombre de valeurs manquantes
lieux_test.drop(columns=['v1', 'v2', 'env1'], inplace=True)

# Résumé des modifications et vérification des valeurs manquantes après nettoyage
final_info_lieux_test = {
    "Info finale": lieux_test.info(),
    "Valeurs manquantes finales": lieux_test.isnull().sum()
}

#print(final_info_lieux_test["Valeurs manquantes finales"])


###VEHICULES
# Application des instructions de nettoyage pour 'vehicules_test'

# Imputation des valeurs manquantes pour 'occutc' avec 0
vehicules_test['occutc'].fillna(0, inplace=True)

# Imputation des valeurs manquantes pour 'motor' avec -1
vehicules_test['motor'].fillna(-1, inplace=True)

# Imputation des valeurs manquantes pour 'senc', 'obs', 'obsm', 'choc', et 'manv' avec -1
for col in ['senc', 'obs', 'obsm', 'choc', 'manv']:
    vehicules_test[col].fillna(-1, inplace=True)

# Vérification finale après le nettoyage pour 'vehicules_test'
vehicules_test_cleaned_info = {
    "Info après nettoyage": vehicules_test.info(),
    "Valeurs manquantes après nettoyage": vehicules_test.isnull().sum()
}

print(vehicules_test_cleaned_info["Valeurs manquantes après nettoyage"])
#id_vehicules bcp sont vides


###USAGERS
# Application des instructions de nettoyage pour 'usagers_test'

# Conversion de 'actp' en catégorique et imputation des valeurs manquantes par -1
usagers_test['actp'] = pd.to_numeric(usagers_test['actp'], errors='coerce')
usagers_test['actp'].fillna(-1, inplace=True)

# Imputation des valeurs manquantes pour 'secu' avec -1
usagers_test['secu'].fillna(-1, inplace=True)
usagers_test['secu1'].fillna(-1, inplace=True)
usagers_test['secu2'].fillna(-1, inplace=True)
usagers_test['secu3'].fillna(-1, inplace=True)

# Imputation de 'an_nais' avec la médiane
an_nais_median = usagers_test['an_nais'].median()
usagers_test['an_nais'].fillna(an_nais_median, inplace=True)

# Traitement des autres colonnes avec valeurs manquantes mineures
for col in ['place', 'trajet', 'locp', 'etatp']:
    usagers_test[col].fillna(-1, inplace=True)

# Résumé final après le nettoyage pour 'usagers_test'
usagers_test_cleaned_info = {
    "Info après nettoyage": usagers_test.info(),
    "Valeurs manquantes après nettoyage": usagers_test.isnull().sum()
}

#print(usagers_test_cleaned_info["Valeurs manquantes après nettoyage"])
#id_vehicules et id_usagers bcp sont vides

