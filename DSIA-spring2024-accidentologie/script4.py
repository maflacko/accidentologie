import pandas as pd

##Nettoyage
def clean_caracteristiques(filepath, delimiter=','):
    df = pd.read_csv(filepath, delimiter=delimiter)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)
    df['adr'].fillna('Inconnue', inplace=True)
    df['atm'].fillna(-1, inplace=True)
    if 'gps' in df.columns:
        df.drop(columns='gps', inplace=True)
    df['hrmn'] = df['hrmn'].apply(lambda x: str(x).replace(':', '')).astype(int)
    df['hour'] = df['hrmn'] // 100
    df['minute'] = df['hrmn'] % 100
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
base_path_train = 'TRAIN/'

# Liste des chemins des fichiers de caractéristiques pour chaque année
carac_file_paths = [
    'TRAIN/BAAC-Annee-2019/caracteristiques_2019_.csv',
    'TRAIN/BAAC-Annee-2020/caracteristiques_2020_.csv',
    'TRAIN/BAAC-Annee-2021/caracteristiques_2021_.csv',
    #'TRAIN/BAAC-Annee-2022/caracteristiques_2022_.csv'
]

# Liste des chemins des fichiers de train pour les lieux, les véhicules et les usagers
lieux_file_paths = [
    'TRAIN/BAAC-Annee-2019/lieux_2019_.csv',
    'TRAIN/BAAC-Annee-2020/lieux_2020_.csv',
    'TRAIN/BAAC-Annee-2021/lieux_2021_.csv',
    #'TRAIN/BAAC-Annee-2022/lieux_2022_.csv'
    # Ajoutez les chemins pour les autres années si nécessaire
]

vehicules_file_paths = [
    'TRAIN/BAAC-Annee-2019/vehicules_2019_.csv',
    'TRAIN/BAAC-Annee-2020/vehicules_2020_.csv',
    'TRAIN/BAAC-Annee-2021/vehicules_2021_.csv',
    #'TRAIN/BAAC-Annee-2021/vehicules_2022_.csv'
    # Ajoutez les chemins pour les autres années si nécessaire
]

usagers_file_paths = [
    'TRAIN/BAAC-Annee-2019/usagers_2019_.csv',
    'TRAIN/BAAC-Annee-2020/usagers_2020_.csv',
    'TRAIN/BAAC-Annee-2021/usagers_2021_.csv',
    #'TRAIN/BAAC-Annee-2022/usagers_2022_.csv'
    # Ajoutez les chemins pour les autres années si nécessaire
]

# Liste des DataFrames correspondant à chaque fichier
caracteristiques_dataframes = [clean_caracteristiques(file_path, delimiter=';') for file_path in carac_file_paths]
# Concaténation des DataFrames en un seul
caracteristiques_train = pd.concat(caracteristiques_dataframes, ignore_index=True)
caracteristiques_test = clean_caracteristiques(base_path_test + 'CARACTERISTIQUES.csv')

#print(caracteristiques_test.dtypes)

#lieux_train = clean_lieux('TRAIN/BAAC-Annee-2019/lieux_2019_.csv', delimiter=';')
lieux_test = clean_lieux(base_path_test + 'LIEUX.csv')

#vehicules_train = clean_vehicules(base_path_train + 'TRAIN/BAAC-Annee-2019/vehicules_2019_.csv',  delimiter=';')
vehicules_test = clean_vehicules(base_path_test + 'VEHICULES.csv')

#usagers_train = clean_usagers(base_path_train + 'TRAIN/BAAC-Annee-2019/usagers_2019_.csv',  delimiter=';')
usagers_test = clean_usagers(base_path_test + 'USAGERS.csv')

# Liste des DataFrames correspondant à chaque fichier de train pour les lieux, les véhicules et les usagers
lieux_dataframes = [clean_lieux(file_path, delimiter=';') for file_path in lieux_file_paths]
vehicules_dataframes = [clean_vehicules(file_path, delimiter=';') for file_path in vehicules_file_paths]
usagers_dataframes = [clean_usagers(file_path, delimiter=';') for file_path in usagers_file_paths]

# Concaténation des DataFrames en un seul pour chaque type de données
lieux_train = pd.concat(lieux_dataframes, ignore_index=True)
vehicules_train = pd.concat(vehicules_dataframes, ignore_index=True)
usagers_train = pd.concat(usagers_dataframes, ignore_index=True)

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

#feature engineering
# After cleaning dataframes
# Place the code right after your data cleaning processes for each DataFrame


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


from sklearn.preprocessing import OneHotEncoder
#print(bdd_test['etatp'].unique())
#print(bdd_train['etatp'].unique())

# Création d'un encodeur One-Hot
#encoder = OneHotEncoder(handle_unknown='ignore')

# Adapter l'encodeur aux données d'entraînement
#encoder.fit(bdd_train[['etatp']])
# Appliquer l'encodage aux données d'entraînement et de test
#bdd_train = encoder.transform(bdd_train[['etatp']])
#bdd_test = encoder.transform(bdd_test[['etatp']])


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

# Séparation des caractéristiques et de la cible
features = ['lat', 'long', 'secu2', 'dep', 'hour', 'minute', 'jour', 'mois', 'Num_Acc', 'agg', 'plan']
X = bdd_train[features]
y = bdd_train['GRAVE']
# Model training with hyperparameter tuning
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=50)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
gbm = GradientBoostingClassifier(random_state=50)
grid_search = GridSearchCV(gbm, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_gbm = grid_search.best_estimator_
y_val_pred = best_gbm.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_pred)
accuracy = accuracy_score(y_val, (y_val_pred > 0.5).astype(int))

print(f"Validation ROC-AUC Score: {roc_auc}")
print(f"Validation Accuracy: {accuracy}")

# Prepare the test set
X_test = bdd_test[features]
test_probabilities = best_gbm.predict_proba(X_test)[:, 1]

# Prepare and save submission file
submission_df = pd.DataFrame({'Num_Acc': bdd_test['Num_Acc'], 'GRAVE': test_probabilities})
submission_df.to_csv('final_submission.csv', index=False)

print(bdd_train.dtypes)
print(bdd_test.dtypes)

print(bdd_train.head())
print(bdd_test.head())