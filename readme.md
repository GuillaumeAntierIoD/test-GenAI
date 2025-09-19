# Démonstrateur de Staging Virtuel par IA

Ce projet est une application web de staging virtuel construite avec Streamlit et propulsée par l'API Gemini de Google. Elle permet de modifier dynamiquement des images d'intérieur en ajoutant, positionnant, harmonisant ou remplaçant des meubles de manière photoréaliste.

## 🚀 Fonctionnalités

L'application propose deux modes principaux pour réaménager une pièce à partir d'une simple image :

### 1. Placer un meuble

- **Bibliothèque de meubles** : Parcourez une bibliothèque de meubles externes via une API, avec pagination et filtrage par catégorie.
- **Ajout à la scène** : Sélectionnez des meubles et ajoutez-les à votre image. L'arrière-plan des objets est automatiquement supprimé.
- **Placement libre** : Utilisez un canevas interactif pour déplacer, redimensionner et faire pivoter les meubles directement sur l'image de la pièce.
- **Harmonisation par IA** : Une fois le placement validé, lancez une harmonisation avec Gemini pour ajuster l'éclairage, la colorimétrie et ajouter des ombres portées réalistes, intégrant parfaitement les nouveaux objets à la scène.

### 2. Remplacer un meuble

- **Remplacement intelligent** : Choisissez un nouveau meuble dans la bibliothèque. L'application utilise la catégorie de cet objet (ex: 'sofa') pour identifier et remplacer automatiquement le meuble correspondant dans l'image originale.
- **Intégration photoréaliste** : L'API Gemini se charge de supprimer l'ancien objet et d'intégrer le nouveau en respectant la perspective, l'échelle et l'éclairage de la pièce.
- **Conservation de la géométrie** : Le prompt est optimisé pour que le nouveau meuble conserve sa propre forme, au lieu de simplement "recouvrir" l'ancien.

### 3. Lien vers le site 



## 🛠️ Technologies utilisées

- **Framework Web** : Streamlit
- **Modèle d'IA Générative** : Google Gemini (via google-generativeai)
- **Manipulation d'images** : Pillow (PIL)
- **Suppression d'arrière-plan** : rembg
- **Canevas interactif** : streamlit-drawable-canvas
- **Requêtes API** : requests
- **Gestion des dépendances** : python-dotenv

## 👤 Auteur

IoD Solutions - Guillaume Antier 