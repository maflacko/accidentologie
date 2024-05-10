import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
#print(caracteristiques.isnull().sum())
#print(lieux.isnull().sum())
#print(vehicules.isnull().sum())



#Analyse des données
print("Statistiques descriptives pour 'caracteristiques':")
print(caracteristiques.describe())

print("\nStatistiques descriptives pour 'lieux':")
print(lieux.describe())

print("\nStatistiques descriptives pour 'usagers':")
print(usagers.describe())

print("\nStatistiques descriptives pour 'vehicules':")
print(vehicules.describe())

# Configuration de l'environnement de visualisation
sns.set(style="whitegrid")

# Histogramme pour 'Conditions d'éclairage'
plt.figure(figsize=(10, 6))
sns.countplot(x='lum', data=caracteristiques)
plt.title('Distribution des Conditions d\'Éclairage lors des Accidents')
plt.xlabel('Conditions d\'Éclairage')
plt.ylabel('Nombre d\'Accidents')
plt.show()

# Histogramme pour 'Conditions atmosphériques'
plt.figure(figsize=(10, 6))
sns.countplot(x='atm', data=caracteristiques)
plt.title('Distribution des Conditions Atmosphériques lors des Accidents')
plt.xlabel('Conditions Atmosphériques')
plt.ylabel('Nombre d\'Accidents')
plt.show()


# Fusion des données sur 'Num_Acc'
combined_data = pd.merge(lieux, usagers, on='Num_Acc', how='left')

# Vérification des premières lignes pour s'assurer de la réussite de la fusion
print(combined_data.head())
# Création du diagramme en boîte pour visualiser la relation entre le type de route et la gravité des accidents
plt.figure(figsize=(10, 6))
sns.boxplot(x='catr', y='grav', data=combined_data)
plt.title('Relation entre le Type de Route et la Gravité des Accidents')
plt.xlabel('Type de Route')
plt.ylabel('Gravité des Accidents')
plt.show()


# Scatter plot pour la relation entre 'lartpc' (largeur du terre-plein central) et 'larrout' (largeur de la route)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='lartpc', y='larrout', data=lieux)
plt.title('Relation entre Largeur du Terre-Plein Central et Largeur de la Route')
plt.xlabel('Largeur du Terre-Plein Central (m)')
plt.ylabel('Largeur de la Route (m)')
plt.show()


# Configurer la taille de la figure et la grille des subplots
plt.figure(figsize=(18, 6))

# Histogramme pour 'lartpc'
#histogramme pr les valeurs continues
#Par exemple, pour la variable lartpc (largeur du terre-plein central) de lieux:
plt.subplot(1, 3, 1)  # 1 ligne, 3 colonnes, position 1
sns.histplot(lieux['lartpc'].dropna(), kde=True, color='blue')
plt.title('Distribution de la Largeur du Terre-Plein Central')

# Boîte à moustaches pour 'larrout'
plt.subplot(1, 3, 2)  # 1 ligne, 3 colonnes, position 2
sns.boxplot(x=lieux['larrout'])
plt.title('Boîte à Moustaches pour la Largeur de la Route')

# Diagramme de dispersion entre 'lartpc' et 'larrout'
plt.subplot(1, 3, 3)  # 1 ligne, 3 colonnes, position 3
sns.scatterplot(x=lieux['lartpc'], y=lieux['larrout'])
plt.title('Relation entre Largeur du Terre-Plein Central et Largeur de la Route')
plt.xlabel('Largeur du Terre-Plein Central (m)')
plt.ylabel('Largeur de la Route (m)')

# Afficher tous les graphiques
plt.tight_layout()  # Ajuste automatiquement les sous-graphiques pour qu'ils s'adaptent à la figure
plt.show()


for col in combined_data.columns:
    print(col)
