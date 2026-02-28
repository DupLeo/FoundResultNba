# TP4 - Distillation de Modèles de Raisonnement (DASD)
Binôme : Léo DUPRAZ-ROGET - Paul COULMEAU

------------------------------------------------------------------------

## 1. Introduction
Le TP visait à :

1.  Générer un dataset de raisonnement via un modèle Teacher
2.  Implémenter un mécanisme de sélection des exemples inspiré du
    Divergence-Aware Sampling (DAS)
3.  Préparer les données au format compatible LLaMA-Factory
4.  Mettre en place une pipeline d'entraînement
5.  Évaluer la faisabilité de la distillation en environnement contraint

------------------------------------------------------------------------

## 2. Génération du Dataset

## 📚 Dataset Kaggle

Pour trouver des données nous avons utilisé le dataset:
- **ID**: eoinamoore/historical-nba-data-and-player-box-scores
- **Données**: Historique NBA avec matchs de 2024-2025 et 2025-2026
- **Source**: https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores

Ce dataset a été choisi car il contient des statistiques riches permettant de construire des prompts de raisonnement complexes pour la prédiction sportive.

Les données ont été importées localement afin d’éviter les problèmes de connexion à l’API Kaggle et pour accélérer le traitement des données.

### 2.1 Construction des prompts

Nous avons construit des prompts structurés à partir des statistiques NBA disponibles.

Les statistiques utilisées incluent :

-   Statistiques classiques (victoires, défaites, points moyens)
-   Statistiques avancées joueurs (PER, TS%, BPM, USG%)
-   Four Factors (eFG%, TOV%, ORB%, FT Rate)

Le script DataNba.py génère automatiquement les prompts d’entraînement.
Principe :
- 30 équipes NBA
- Chaque équipe peut jouer contre les 29 autres équipes
- Génération de statistiques moyennes sur les 10 derniers matchs

Cela permet de générer plus de 800 prompts d’entraînement.

Format des Prompts Teacher :

```text
      SYSTEM:
      Tu es un modèle d'IA qui répond UNIQUEMENT en JSON.
      Ne donne aucune explication.
      Ne fais aucune phrase.
      Ne mets pas de texte avant ou après le JSON.

      Format strict :

      {
      "equipe_gagnante": "string",
      "score_predit": "string",
      "confiance": number
      }

      USER:

      Analyse ce match NBA.

      ÉQUIPE 1: {team1_stats.get('team')}
      - Victoires: {team1_stats.get('victoires')}
      - Défaites: {team1_stats.get('defaites')}
      - Points moyens: {team1_stats.get('points_moyens'):.1f}

      Stats avancées:
      {team1_adv}

      Four Factors:
      {team1_four}

      ---

      ÉQUIPE 2: {team2_stats.get('team')}
      - Victoires: {team2_stats.get('victoires')}
      - Défaites: {team2_stats.get('defaites')}
      - Points moyens: {team2_stats.get('points_moyens'):.1f}

      Stats avancées:
      {team2_adv}

      Four Factors:
      {team2_four}

      Contexte:
      Lieu: {match_info.get('lieu')}
      Date: {match_info.get('date')}
      Saison: {match_info.get('saison')}

      Réponds uniquement avec le JSON.
```
------------------------------------------------------------------------

### 2.2 Format des données

Les données générées ont été converties au format ShareGPT compatible
avec LLaMA-Factory :

```json
{ 
  "conversations":
    [
      { "from": "human", "value": "Prompt NBA complet"},
      { "from": "assistant", "value": "Réponse JSON du Teacher" }
    ]
}
```
### 2.3 Model Teacher 

Nous avons utilisé le modèle ``meta-llama/Meta-Llama-3-8B-Instruct`` pour générer les réponses du Teacher car l'API infomaniak ne fonctionnait pas après de nombreux essais.
Disponible sur Hugging Face : https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

------------------------------------------------------------------------

## 3. Tentative d'implémentation du Divergence-Aware Sampling (DAS)

### 3.1 Principe théorique

Le DAS repose sur :

-   L'analyse des log-probabilités token par token
-   La comparaison Teacher vs Student
-   L'identification de Teacher Sentences

L'idée est de conserver les exemples où le Teacher est confiant et le Student incertain.
De cette manière, il est possible d'améliorer la rapidité et l'efficacité au moment de 
l'entrainement.


------------------------------------------------------------------------

### 3.2 Problème rencontré : logprobs 

Dans le fichier ``DASPipeline`` la fonction ``compute_quality_score``, nous calculons un score de qualité basé sur la divergence entre les log-probabilités du Teacher et du Student. Nous avons tenté d'exploiter l'option ``logprobs=True`` lors des appels au modèle Teacher. Malheureusement, le model ne fournissait pas toujours de log-probabilités des tokens.

------------------------------------------------------------------------

### 3.3 Conséquence

Nous avons pu implémenter cette solution. Mais faute d'accès aux log-probs, nous n'avons pas pu l'exploiter pour filtrer les exemples d'entraînement. Nous avons donc utilisé l'ensemble complet des données générées par le Teacher pour entraîner le Student.

------------------------------------------------------------------------

## 4. Préparation de l'Entraînement

Les données ont été :

1.  Nettoyées (disponible sur hugging face : https://huggingface.co/datasets/yann756/distillation-nba)
2.  Structurées en ShareGPT
3.  Séparées en jeux compatibles Stage 1 / Stage 2

L'entraînement prévu utilisait :

-   Fine-tuning LoRA
-   Chargement du modèle Student en 4-bit
-   Adaptateurs successifs

------------------------------------------------------------------------

## 5. Conclusion

Ce TP a permis d'explorer concrètement :

-   La distillation de raisonnement
-   La structuration de datasets supervisés
-   Les limites pratiques des APIs LLM

Bien que l'implémentation complète du DAS n'ait pas été complète, nous avons pu manipuler
la distillation et l'architecture autour de ce processus.