import pandas as pd
from sklearn.preprocessing import LabelEncoder

##Nettoyage
def clean_caracteristiques(filepath, delimiter=','):
    df = pd.read_csv(filepath, delimiter=delimiter)
    # Suppression d'une colonne non désirée si elle existe
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)
    
    df['adr'].fillna('Inconnue', inplace=True)
    df['atm'].fillna(-1, inplace=True)
    if 'gps' in df.columns:
        df.drop(columns='gps', inplace=True)

    # Mode et imputation pour 'hrmn'
    #df.drop(columns='hrmn', inplace=True)
#    hrmn_mode = df['hrmn'].mode()
#    if not hrmn_mode.empty:
#        df['hrmn'].fillna(hrmn_mode[0], inplace=True)
#    else:
#        df['hrmn'].fillna(pd.Timestamp('00:00').time(), inplace=True)

    df['hrmn'] = df['hrmn'].apply(lambda x: str(x).replace(':', ''))

    df['lat'] = df['lat'].astype(str).str.replace(',', '.').astype(float)
    df['long'] = df['long'].astype(str).str.replace(',', '.').astype(float)
    df['lat'].fillna(0.0, inplace=True)
    df['long'].fillna(0.0, inplace=True)

    df['dep'] = df['dep'].apply(lambda x: '0' if x in ['2A', '2B', '2B033'] else x)

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
    if 'id_vehicule' in df.columns:
        df['id_vehicule'] = df['id_vehicule'].apply(lambda x: ''.join(str(x).split()).replace('.0', ''))   
    return df

def clean_usagers(filepath, delimiter=','):
    df = pd.read_csv(filepath,  delimiter=delimiter)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)
    if 'secu' in df.columns:
        df.drop(columns='secu', inplace=True)
    if 'id_usager' in df.columns:
        df.drop(columns='id_usager', inplace=True) 
    if 'id_vehicule' in df.columns:
        df['id_vehicule'] = df['id_vehicule'].apply(lambda x: ''.join(str(x).split()).replace('.0', ''))    
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

caracteristiques_train = clean_caracteristiques(base_path_train + 'caracteristiques_2019_.csv', delimiter=';')
caracteristiques_test = clean_caracteristiques(base_path_test + 'CARACTERISTIQUES.csv')

#print(caracteristiques_test.dtypes)

lieux_train = clean_lieux(base_path_train + 'lieux_2019_.csv', delimiter=';')
lieux_test = clean_lieux(base_path_test + 'LIEUX.csv')

vehicules_train = clean_vehicules(base_path_train + 'vehicules_2019_.csv',  delimiter=';')
vehicules_test = clean_vehicules(base_path_test + 'VEHICULES.csv')

usagers_train = clean_usagers(base_path_train + 'usagers_2019_.csv',  delimiter=';')
usagers_test = clean_usagers(base_path_test + 'USAGERS.csv')


#print(vehicules_train.head())
#print(vehicules_test.head())
#print(vehicules_test.isnull().sum())


#print(vehicules_train.dtypes)
#print(vehicules_test.dtypes)
#print(usagers_test.dtypes)

#print(vehicules_test.isnull().sum())


# Appliquer le nettoyage à chaque DataFrame
#vehicules_test = clean_id_vehicule(vehicules_test)
#usagers_test = clean_id_vehicule(usagers_test)

# Filtrer pour obtenir les DataFrames où Num_Acc est 201900004650
#filtered_vehic = vehicules_test[vehicules_test['Num_Acc'] == 201900004650]
#filtered_usag = usagers_test[usagers_test['Num_Acc'] == 201900004650]

# Afficher les valeurs de id_vehicule pour ces lignes spécifiques
#print("ID Véhicule pour Num_Acc = 201900004650 dans véhicules:")
#print(filtered_vehic['id_vehicule'])
#print("ID Véhicule pour Num_Acc = 201900004650 dans usagers:")
#print(filtered_usag['id_vehicule'])


#fusion train
bdd_train = pd.merge(caracteristiques_train, lieux_train, on='Num_Acc', how='outer')
bdd_train = pd.merge(bdd_train, vehicules_train, on='Num_Acc', how='outer')
bdd_train = pd.merge(bdd_train, usagers_train, on=['Num_Acc', 'id_vehicule', 'num_veh'], how='outer')

#fusion test
bdd_test = pd.merge(caracteristiques_test, lieux_test, on='Num_Acc', how='outer')
bdd_test = pd.merge(bdd_test, vehicules_test, on='Num_Acc', how='outer')
bdd_test = pd.merge(bdd_test, usagers_test, on=['Num_Acc', 'id_vehicule', 'num_veh'], how='outer')

bdd_test = bdd_test.dropna(subset=['an_nais'])

#print(bdd_train.head())
#print(bdd_train.shape[0])
#print(bdd_test.head())


#print("VERIF")
#print(bdd_test.isnull().sum())

#print(bdd_test.shape[0])


bdd_test.to_csv('test.csv', index=False)

# Creating the target variable 'GRAVE'
# GRAVE = 1 if at least one user involved in the accident has grav = 2 or 3, and 0 otherwise.

# First, group the data by 'Num_Acc' and check if any of the users involved has grav = 2 or 3
bdd_train['is_grave'] = bdd_train['grav'].apply(lambda x: 1 if x in [2, 3] else 0)

grave_by_accident = bdd_train.groupby('Num_Acc')['is_grave'].max()
# Joindre cette information à votre DataFrame original
bdd_train = bdd_train.join(grave_by_accident, on='Num_Acc', rsuffix='_final')

# Renommez la colonne comme nécessaire
bdd_train.rename(columns={'is_grave_final': 'GRAVE'}, inplace=True)
bdd_train.drop('is_grave', axis=1, inplace=True)

#ANALYSE EXPLORATOIRE

import matplotlib.pyplot as plt
import seaborn as sns

# Résumé statistique des données
summary_stats = bdd_train.describe()

# Distribution de la variable cible 'GRAVE'
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(x='GRAVE', data=bdd_train, ax=ax[0])
ax[0].set_title('Distribution of GRAVE')
ax[0].set_xlabel('GRAVE Status')
ax[0].set_ylabel('Count')

# Relation entre la lumière (lum) et GRAVE
sns.countplot(x='lum', hue='GRAVE', data=bdd_train, ax=ax[1])
ax[1].set_title('Impact of Lighting Condition on GRAVE Status')
ax[1].set_xlabel('Lighting Condition')
ax[1].set_ylabel('Count')

plt.tight_layout()
#plt.show()

# Afficher le résumé statistique pour observer d'autres caractéristiques potentiellement intéressantes
print(summary_stats)


# Calcul de la matrice de corrélation pour toutes les variables numériques fournies
all_columns = [
    'Num_Acc', 'jour', 'mois', 'an', 'lum', 'dep', 'com', 'agg', 'int', 'atm', 'col',
    'adr', 'lat', 'long', 'catr', 'circ', 'nbv', 'vosp', 'prof', 'plan', 'lartpc', 'larrout', 'surf',
    'infra', 'situ', 'vma', 'id_vehicule', 'num_veh', 'senc', 'catv', 'obs', 'obsm', 'choc', 'manv',
    'motor', 'occutc', 'place', 'catu', 'grav', 'sexe', 'an_nais', 'trajet', 'secu1', 'secu2', 'secu3',
    'locp', 'actp', 'etatp', 'GRAVE'
]

# Filtrer le dataframe pour ne garder que les colonnes numériques
numerical_data = bdd_train[all_columns].select_dtypes(include=['float64', 'int64'])

# Calcul de la matrice de corrélation pour les variables numériques sélectionnées
full_corr_matrix = numerical_data.corr()

# Affichage de la matrice de corrélation complète avec un heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(full_corr_matrix, annot=False, cmap='coolwarm', linewidths=.5)
plt.title('Heatmap of Full Correlation Matrix')
#plt.show()

# Identifier les corrélations significatives avec 'GRAVE'
significant_correlations = full_corr_matrix['GRAVE'].sort_values(key=abs, ascending=False)

# Afficher les variables qui ont une corrélation supérieure à un seuil significatif (par exemple 0.05)
significant_vars = significant_correlations[abs(significant_correlations) > 0.05]
print(significant_vars)


#PRETRAITEMENT DES DONNEES (valeurs manquantes + encodage si necessaire)
#print(bdd_train.isnull().sum())

#Les seules nouvelles lignes vides correspondent à un manque d'enregistrement
# par ex 1 accident, 2 vehicules et 1 seul usager, la deuxieme ligne vehicule aura des donnés manquantes sur un usager
#on peut donc supprimer/ignoer ces lignes car c usagers qui porte la variable grav
# Supprimer les lignes où 'grav' est NaN
bdd_train = bdd_train.dropna(subset=['grav'])

#suppression des colonnes non pertinentes à l'entrainement modèle
bdd_train.drop(columns=['grav', 'id_vehicule', 'com', 'adr', 'num_veh'], inplace=True)
bdd_test.drop(columns=['id_vehicule', 'com', 'adr', 'num_veh'], inplace=True)

bdd_train.to_csv('train_final.csv', index=False)
bdd_test.to_csv('test_final.csv', index=False)

from sklearn.model_selection import train_test_split

# Séparation des caractéristiques et de la cible
X = bdd_train.drop(columns=['GRAVE'])
y = bdd_train['GRAVE']

# Division de l'ensemble d'entraînement en sous-ensembles d'entraînement et de validation (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=50)

# Affichage des dimensions des ensembles pour vérifier
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

from sklearn.ensemble import RandomForestClassifier

# Création du modèle de forêts aléatoires
rf_model = RandomForestClassifier(n_estimators=100, random_state=50)

# Entraînement du modèle
rf_model.fit(X_train, y_train)

# Obtention de l'importance des caractéristiques
feature_importances = rf_model.feature_importances_

# Création d'un DataFrame pour mieux visualiser les importances
importances_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(importances_df.head(10))  # Affichage des 10 caractéristiques les plus importantes

# Sélection des 10 caractéristiques principales
top_features = importances_df['Feature'].head(10).tolist()

# Création de nouveaux sous-ensembles de données avec seulement les 10 caractéristiques principales
X_train_top = X_train[top_features]
X_val_top = X_val[top_features]

# Entraînement du modèle de forêts aléatoires avec seulement les 10 caractéristiques principales
rf_model_top = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_top.fit(X_train_top, y_train)

# Evaluation du modèle sur l'ensemble de validation
from sklearn.metrics import accuracy_score, roc_auc_score

y_val_pred = rf_model_top.predict(X_val_top)
accuracy_top = accuracy_score(y_val, y_val_pred)
roc_auc_top = roc_auc_score(y_val, y_val_pred)

accuracy_top, roc_auc_top


# Préparation de l'ensemble de test avec les mêmes 10 caractéristiques principales
X_test_top = bdd_test[top_features]

# Prédiction des probabilités pour l'ensemble de test
test_probabilities = rf_model_top.predict_proba(X_test_top)[:, 1]

# Préparation du fichier de soumission
submission_df = pd.DataFrame({
    'Num_Acc': bdd_test['Num_Acc'],
    'GRAVE': test_probabilities
})

# Vérification des premières lignes du fichier de soumission
print(submission_df.head())

# Calcul de la probabilité maximale pour chaque identifiant d'accident unique
final_submission_df = submission_df.groupby('Num_Acc')['GRAVE'].max().reset_index()

# Vérification des premières lignes du fichier de soumission final
final_submission_df.head()
# Enregistrement du DataFrame dans un fichier CSV avec une précision à quatre décimales pour les probabilités
final_submission_path = 'final_submission.csv'
final_submission_df.to_csv(final_submission_path, index=False)



