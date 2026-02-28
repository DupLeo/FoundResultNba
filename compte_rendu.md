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

### 2.1 Construction des prompts

Nous avons construit des prompts à partir d'un dataset NBA
contenant :

-   Statistiques classiques (victoires, défaites, points moyens)
-   Statistiques avancées joueurs (PER, TS%, BPM, USG%)
-   Four Factors (eFG%, TOV%, ORB%, FT Rate)

Chaque prompt demandait au Teacher :

-   D'analyser les statistiques
-   De prédire le vainqueur
-   De répondre strictement au format JSON que nous avions choisi

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

### 3.2 Problème rencontré : logprobs indisponibles

Nous avons tenté d'exploiter l'option ``logprobs=True`` lors des appels au
modèle Teacher.
Mais malheureusement les modèles auxquels nous avions accès ne fournissaient pas les
log-probabilités des tokens générés. Puis l'API infomaniak ne fonctionnait pas après
de nombreux essais.

Après de longues recherches certaines APIs ne supportaient pas logprobs=True, et les 
modèles Hugging Face accessibles ne proposaient pas directement cette fonctionnalité.

------------------------------------------------------------------------

### 3.3 Conséquence

Nous n'avons pas pu implémenter cette solution. Nous avons donc opté pour la solution
simple : ne pas les utiliser.
Cela aurait permis de gagner en temps d'entrainement, mais il n'est pas obligatoire 
pour avoir de bons résultats.

------------------------------------------------------------------------

## 4. Préparation de l'Entraînement

Les données ont été :

1.  Nettoyées
2.  Et structurées

L'entraînement prévu utilisait :

-   Fine-tuning LoRA
-   Chargement du modèle Student en 4-bit
-   Adaptateurs successifs

Les explications précises de l'entrainement sont disponibles dans le
README du dossier ``/training``.
Le modèle est disponible à l'adresse suivante : https://huggingface.co/pollord/qwen_nba

------------------------------------------------------------------------

## 5. Conclusion

Ce TP a permis d'explorer concrètement :

-   La distillation de raisonnement
-   La structuration de datasets supervisés
-   Les limites pratiques des APIs LLM

Bien que l'implémentation complète du DAS n'ait
pas été possible faute d'accès aux logprobs, nous avons pu manipuler
la distillation et l'architecture autour de ce processus.