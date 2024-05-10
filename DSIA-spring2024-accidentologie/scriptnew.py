import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



base_path_train = 'TRAIN/BAAC-Annee-2019/'

caracteristiques = pd.read_csv(base_path_train + 'caracteristiques_2019_.csv', sep=';')
lieux = pd.read_csv(base_path_train + 'lieux_2019_.csv', sep=';')
usagers = pd.read_csv(base_path_train + 'usagers_2019_.csv', sep=';')
vehicules = pd.read_csv(base_path_train + 'vehicules_2019_.csv', sep=';')

caracteristiques.drop(columns=['Unnamed: 0'], inplace=True)
lieux.drop(columns=['Unnamed: 0'], inplace=True)
usagers.drop(columns=['Unnamed: 0'], inplace=True)
vehicules.drop(columns=['Unnamed: 0'], inplace=True)

print(usagers.isnull().sum())

# Correction de l'imputation en utilisant la bonne colonne
caracteristiques['adr'].fillna('Inconnue', inplace=True)  # Imputer 'adr' avec 'Inconnue'
lieux['voie'].fillna('Non spécifiée', inplace=True)
# Assumer que 'v1' est 0 là où l'information est manquante, si cela est logique selon le contexte
lieux['v1'].fillna(0, inplace=True)
lieux['v2'].fillna('Non spécifié', inplace=True)
# Assumer que 'lartpc' et 'larrout' sont 0 s'il n'y a pas d'information
lieux['lartpc'].fillna(0, inplace=True)
lieux['larrout'].fillna(0, inplace=True)
# Assumer que 'occutc' est 0 s'il n'y a pas d'information dans le DataFrame 'vehicules'
vehicules['occutc'].fillna(0, inplace=True)
# Conversion des colonnes 'lat' et 'long' en float dans le dataframe caracteristiques
caracteristiques['lat'] = caracteristiques['lat'].str.replace(',', '.').astype(float)
caracteristiques['long'] = caracteristiques['long'].str.replace(',', '.').astype(float)



#print(caracteristiques.isnull().sum())
#print(lieux.isnull().sum())
#print(vehicules.isnull().sum())
#print(usagers.isnull().sum())

#FUSION bdd
# Merge caracteristiques and lieux
merged_data = pd.merge(caracteristiques, lieux, on='Num_Acc', how='outer')
# Merge the result with vehicules
merged_data = pd.merge(merged_data, vehicules, on='Num_Acc', how='outer')
# Merge the result with usagers
# Avant de fusionner avec usagers, assure-toi que id_vehicule et num_veh sont utilisés correctement
# Assure-toi que les colonnes id_vehicule et num_veh sont bien nommées et consistentes dans tous les DataFrames
merged_data = pd.merge(merged_data, usagers, on=['Num_Acc', 'id_vehicule', 'num_veh'], how='outer')
# Création de la variable cible GRAVE
merged_data['GRAVE'] = merged_data['grav'].apply(lambda x: 1 if x in [2, 3] else 0)

# Show the first few rows of the fully merged dataframe to verify the structure
print(merged_data.info())
print(merged_data.head())

for col in merged_data.columns:
    print(col)

#après analyse de la fusion, si la colonne 'grav' n'est jamais nulle dans le dataframe usagers original, mais que des valeurs nulles apparaissent après la fusion, cela suggère que certaines lignes des autres dataframes (comme vehicules ou caracteristiques) ne correspondent pas parfaitement à usagers. Les valeurs nulles pourraient résulter de lignes dans vehicules ou caracteristiques qui n'ont pas de correspondance directe dans usagers, menant à des fusions incomplètes.
# Supprimer les lignes où 'grav' est nul dans le dataframe fusionné
merged_data = merged_data.dropna(subset=['grav'])
# Vérifier la taille du dataframe après suppression
print(merged_data.shape)


###Transformation des données

# Convertir 'hrmn' en minutes depuis minuit
#merged_data['hrmn'] = data_merged['hrmn'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if pd.notna(x) else np.nan)



# Selecting features - for now, we'll select a subset that might seem relevant
#features = merged_data.drop(columns=['grav','GRAVE', 'Num_Acc', 'id_vehicule', 'num_veh', 'an', 'an_nais', 'hrmn', 'dep', 'com', 'adr', 'voie','v2', 'actp', 'pr', 'pr1'])  # excluding identifiers and the target variable
features = merged_data[['jour', 'mois', 'lum', 'agg', 'int', 'atm', 'col', 'lat', 'long', 'catr', 'v1', 'circ', 'nbv', 'vosp', 'prof', 'plan', 'lartpc', 'larrout', 'surf', 'infra', 'situ', 'vma', 'senc', 'catv', 'obs', 'obsm', 'choc', 'manv', 'motor', 'occutc', 'place', 'catu', 'sexe', 'trajet', 'secu1', 'secu2', 'secu3', 'locp', 'etatp']]
print(merged_data.isnull().sum())
# Define the target variable again
target = merged_data['GRAVE'] 


#analyse
import matplotlib.pyplot as plt

# Analyser la distribution de la variable cible 'grav'
plt.figure(figsize=(8, 6))
merged_data['grav'].value_counts().plot(kind='bar')
plt.title('Distribution de la Gravité des Accidents')
plt.xlabel('Gravité')
plt.ylabel('Nombre d\'Accidents')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

import seaborn as sns

# Analyser la corrélation entre les features et la variable cible
correlation_matrix = merged_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matrice de Corrélation')
#plt.show()



# Split the data into train and test sets once more
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42) #42??

# Display the shapes of the train and test datasets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(merged_data.info())

from sklearn.ensemble import RandomForestClassifier

# Instanciation du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement du modèle
model.fit(X_train, y_train)

# Évaluation sur l'ensemble de test
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Tu peux également prédire de nouvelles données
# new_data_predictions = model.predict(new_data)



###TEST

base_path_test = 'TEST/TEST/'

caracteristiques_test = pd.read_csv(base_path_test + 'CARACTERISTIQUES.csv')
lieux_test = pd.read_csv(base_path_test + 'LIEUX.csv')
usagers_test = pd.read_csv(base_path_test + 'USAGERS.csv')
vehicules_test = pd.read_csv(base_path_test + 'VEHICULES.csv')



# Correction de l'imputation en utilisant la bonne colonne
caracteristiques_test['adr'].fillna('Inconnue', inplace=True)  # Imputer 'adr' avec 'Inconnue'
lieux_test['voie'].fillna('Non spécifiée', inplace=True)
# Assumer que 'v1' est 0 là où l'information est manquante, si cela est logique selon le contexte
lieux_test['v1'].fillna(0, inplace=True)
lieux_test['v2'].fillna('Non spécifié', inplace=True)
# Assumer que 'lartpc' et 'larrout' sont 0 s'il n'y a pas d'information
lieux_test['lartpc'].fillna(0, inplace=True)
lieux_test['larrout'].fillna(0, inplace=True)
# Assumer que 'occutc' est 0 s'il n'y a pas d'information dans le DataFrame 'vehicules'
vehicules_test['occutc'].fillna(0, inplace=True)
# Conversion des colonnes 'lat' et 'long' en float dans le dataframe caracteristiques
#pas la même chose que train ATTENTION
# Convertir les données en chaînes si ce n'est pas déjà le cas
caracteristiques_test['lat'] = caracteristiques_test['lat'].astype(str)
caracteristiques_test['long'] = caracteristiques_test['long'].astype(str)
# Remplacer les virgules par des points et convertir en float
caracteristiques_test['lat'] = caracteristiques_test['lat'].str.replace(',', '.').astype(float)
caracteristiques_test['long'] = caracteristiques_test['long'].str.replace(',', '.').astype(float)
caracteristiques_test['atm'].fillna(-1, inplace=True)
caracteristiques_test['lat'].fillna(-999, inplace=True)
caracteristiques_test['long'].fillna(-999, inplace=True)


lieux_test['lartpc'] = lieux_test['lartpc'].astype(str)
lieux_test['larrout'] = lieux_test['larrout'].astype(str)
# Remplacer les virgules par des points et convertir en float
lieux_test['lartpc'] = lieux_test['lartpc'].str.replace(',', '.').astype(float)
lieux_test['larrout'] = lieux_test['larrout'].str.replace(',', '.').astype(float)
lieux_test['circ'].fillna(-1, inplace=True)
lieux_test['nbv'].fillna(-1, inplace=True)
lieux_test['vosp'].fillna(-1, inplace=True)
lieux_test['prof'].fillna(-1, inplace=True)
lieux_test['plan'].fillna(-1, inplace=True)
lieux_test['surf'].fillna(-1, inplace=True)
lieux_test['infra'].fillna(-1, inplace=True)
lieux_test['situ'].fillna(-1, inplace=True)
lieux_test['vma'].fillna(-1, inplace=True)

vehicules_test['senc'].fillna(-1, inplace=True)
vehicules_test['obs'].fillna(-1, inplace=True)
vehicules_test['obsm'].fillna(-1, inplace=True)
vehicules_test['choc'].fillna(-1, inplace=True)
vehicules_test['manv'].fillna(-1, inplace=True)
vehicules_test['motor'].fillna(-1, inplace=True)

usagers_test['place'].fillna(10, inplace=True)
usagers_test['trajet'].fillna(-1, inplace=True)
usagers_test['secu1'].fillna(-1, inplace=True)
usagers_test['secu2'].fillna(-1, inplace=True)
usagers_test['secu3'].fillna(-1, inplace=True)

usagers_test['locp'].fillna(-1, inplace=True)
usagers_test['etatp'].fillna(-1, inplace=True)


#columns_to_check = ['jour', 'mois', 'lum', 'agg', 'int', 'atm', 'col', 'lat', 'long', 'catr', 'v1', 'circ', 'nbv', 'vosp', 'prof', 'plan', 'lartpc', 'larrout', 'surf', 'infra', 'situ', 'vma', 'senc', 'catv', 'obs', 'obsm', 'choc', 'manv', 'motor', 'occutc', 'place', 'catu', 'sexe', 'trajet', 'secu1', 'secu2', 'secu3', 'locp', 'etatp']


# Vérification après nettoyage
print("VERIF")
print(caracteristiques_test.isnull().sum())
print(lieux_test.isnull().sum())
print(vehicules_test.isnull().sum())
print(usagers_test.isnull().sum())


# Appliquer le même nettoyage de données que pour l'ensemble d'entraînement

merged_data_test = pd.merge(caracteristiques_test, lieux_test, on='Num_Acc', how='outer')
merged_data_test = pd.merge(merged_data_test, vehicules_test, on='Num_Acc', how='outer')
#pas la même chose que train ATTENTION
merged_data_test = pd.merge(merged_data_test, usagers_test, on='Num_Acc', how='outer')



test_features = merged_data_test[features.columns]  # Utilise les mêmes noms de colonnes que dans 'features'
print("VERIF NULL")
print(test_features.isnull().sum())
#### Prédiction de `grav`
predicted_grav = model.predict(test_features)  # Assurez-vous que le modèle est bien entraîné et disponible

### Création de `GRAVE` basée sur les prédictions de `grav`
merged_data_test['grav'] = predicted_grav  # Ajout de la prédiction au DataFrame
merged_data_test['GRAVE'] = merged_data_test['grav'].apply(lambda x: 1 if x in [2, 3] else 0)

### Création du fichier de soumission
submission = pd.DataFrame({
    'Num_Acc': merged_data_test['Num_Acc'],
    'GRAVE': merged_data_test['grav']
})
submission.to_csv('submission.csv', index=False)

