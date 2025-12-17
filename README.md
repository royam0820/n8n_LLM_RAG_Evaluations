## Maîtriser l’Évaluation des LLM et des Workflows dans n8n

Ce dépôt accompagne mon meetup n8n sur le thème **« Maîtriser l’Évaluation des LLM et Workflows dans n8n »**.  
Il regroupe les supports de présentation, les workflows n8n, les jeux de données de référence (gold standard) et des scripts Python pour évaluer vos workflows à l’échelle (avec l’API Google Gemini).

---

## Contenu du dépôt

- **Présentation principale**
  - `Evaluation workflow n8n.pdf`  
  - Slides utilisées pendant le meetup. Elles expliquent les concepts clés :
    - Pourquoi et comment évaluer un LLM / un workflow n8n  
    - Métriques d’évaluation et cas d’usage  
    - Démo pas‐à‐pas avec les fichiers de ce dépôt

- **`EVALUATION/REFERENTIEL` – Jeux de données « gold standard »**
  - `1-QA_Evaluations.csv`  
  - `2-QA_LuminaCorp.csv`  
  - **Rôle**: ensembles de données de référence servant de vérité terrain pour comparer les réponses produites par vos workflows n8n / LLM.  
  - **Usage typique**:
    - Colonnes avec *prompt / question*  
    - Colonnes avec *réponse attendue / label de qualité*  
    - Utilisés par les workflows n8n et le script Python pour calculer des scores d’évaluation.

- **`EVALUATION/JSON_QA_Evalutions` – Workflows n8n (JSON)**
  - Sous-dossier `Historic_Events/`  
    - `1-Loading Reference Tables.json`  
    - `2-Chat Message to LLM Workflow.json`  
    - `3-Chat Message to LLM Workflow - Pirate Edition.json`  
    - `4-Export QA Evaluations Data to Google Sheet (with Pirate Detection).json`  
    - `5-📊 Evaluation Viz.json`  
  - Sous-dossier `Lumina Corp (RAG)/`
    - `App_RAG Agent - Advanced Citation Evaluation.json`  
  - **Rôle**: workflows n8n prêts à l’emploi pour :
    - Charger les jeux de données de référence
    - Appeler un LLM (incluant une version « Pirate Edition » pour montrer la détection de dérives de style)
    - Exporter les résultats d’évaluation (par ex. vers Google Sheets)
    - Visualiser les métriques d’évaluation
    - Évaluer un agent RAG (LuminaCorp) avec vérification avancée des citations
  - **Import dans n8n**:
    1. Ouvrir votre instance n8n.
    2. Créer un nouveau workflow.
    3. Utiliser **Import from file** et choisir le fichier `.json` souhaité.
    4. Mettre à jour les credentials (LLM, Google, etc.) selon votre environnement.

- **`EVALUATION/PYTHON` – Évaluation à l’échelle avec Python & Google Gemini**
  - `rag_evaluator.py`  
  - `RAG_Evaluation_Gemini.ipynb`  
  - **Rôle**: permettre une évaluation massive (jusqu’à ~1500 évaluations gratuites / jour) grâce à l’API **Google Gemini**.
  - **Scénario d’usage recommandé**:
    1. Ouvrir le notebook `RAG_Evaluation_Gemini.ipynb` dans Google Colab ou localement.  
    2. Configurer votre clé d’API Gemini (variable d’environnement ou cellule dédiée dans le notebook).  
    3. Charger un jeu de données depuis le dossier `REFERENTIEL`.  
    4. Appeler les fonctions de `rag_evaluator.py` pour:
       - Exécuter les prompts/questions
       - Comparer les réponses du modèle à la vérité terrain
       - Calculer et exporter des métriques agrégées.
  - **Objectif**: compléter l’évaluation effectuée dans n8n par une approche code (Python) pour:
    - Lancer des campagnes d’évaluation volumineuses
    - Automatiser des rapports qualité
    - Tester rapidement plusieurs variantes de prompts / workflows.

- **`EVALUATION/EVALUATION_Results` – Résultats d’évaluation**
  - `1-QA_Evaluations_output.csv`  
  - `2-QA_LuminaCorp_output.csv`  
  - `3-QA_Evaluations_pirate_output.csv`  
  - `4-QA_LuminaCorp_python_eval.csv`  
  - **Rôle**: exemples de sorties générées par les workflows n8n et par le script Python, incluant :
    - Scores d’évaluation
    - Comparaison entre réponses attendues et réponses LLM
    - Détection de dérives (ex. style « pirate »)

---

## Comment reproduire la démo du meetup

- **1. Explorer la présentation**
  - Ouvrir `Evaluation workflow n8n.pdf` pour une vue d’ensemble des concepts et du scénario démo.

- **2. Importer les workflows dans n8n**
  - Importer les fichiers du dossier `EVALUATION/JSON_QA_Evalutions` dans votre instance n8n.
  - Configurer vos credentials (LLM, Google Sheets, etc.).
  - Lancer les workflows pour:
    - Générer des réponses avec le LLM
    - Calculer des évaluations QA
    - Exporter les résultats (par ex. vers Google Sheets).

- **3. Lancer une évaluation à l’échelle avec Python & Gemini**
  - Ouvrir `RAG_Evaluation_Gemini.ipynb` (de préférence dans Google Colab).  
  - Renseigner votre clé API Gemini.  
  - Utiliser les CSV du dossier `REFERENTIEL` comme données d’entrée.  
  - Lancer les cellules qui appellent `rag_evaluator.py` pour effectuer une campagne d’évaluation plus large.

---

## Objectif pédagogique

Ce dépôt est conçu pour vous aider à :

- **Comprendre** les enjeux de l’évaluation des LLM et des workflows n8n.  
- **Mettre en pratique** via des workflows n8n concrets (chargement de référentiels, génération, scoring, visualisation).  
- **Passer à l’échelle** grâce à un script Python et à l’API Gemini pour industrialiser vos évaluations.  

N’hésitez pas à cloner le dépôt, adapter les jeux de données à vos propres cas d’usage et modifier les workflows / scripts pour vos besoins en production.


