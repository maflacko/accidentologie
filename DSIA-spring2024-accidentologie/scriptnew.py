import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



base_path_train = 'TRAIN/BAAC-Annee-2019/'
#data_test = pd.read_csv('chemin_vers_le_fichier_test.csv')

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
features = merged_data.drop(columns=['grav', 'Num_Acc', 'id_vehicule', 'num_veh', 'an', 'an_nais', 'hrmn', 'dep', 'com', 'adr', 'voie','v2', 'actp', 'pr', 'pr1'])  # excluding identifiers and the target variable
#features = merged_data[['jour', 'mois', 'lum', 'agg', 'int', 'atm', 'col', 'catr', 'circ', 'surf', 'infra', 'situ', 'vma']]
print(merged_data.isnull().sum())
# Define the target variable again
target = merged_data['grav'] 


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
plt.show()



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
