# TRAINING

Dans ce dossier, il y a le fichier de config qui a servi à
entrainer le modèle puis à push le LoRA sur Hugging Face.

Pour l'entrainer, nous avons suivi ces étapes :

## LLamaFactory
Nous avons cloné le projet LLamaFactory dans notre projet,
puis à l'aide de la commande ``pip install e .`` nous avons
installé les dépendances pour lancer correctement le training.

## Conversion
Nous avons ensuite converti les données JSON vers le format
``alpaca`` qui est un des formats supporté par LLamaFactory.

## Configuration
Nous avons créé un fichier de config en .yaml qui permet de 
spécifier les différents élements pour l'entrainement du 
modèle à partir des données que nous avons converti plus tôt.
Il est disponible dans ce dossier avec les informations comme
le modèle de base, les données à utiliser, le format, le nombre
d'epochs, learning rate, etc...

## Datas
Maintenant le fichier de config crée, il faut placer le fichier
de données dans le fichier ``/data`` du repo LLamaFactory. De
cette manière, il sera possible d'être reconnu au moment du lancement.

## Spécification des données d'entrées
Dans le fichier ``dataset_info.json`` nous avons ajouté ces lignes :
```json
  "nba_dataset": {
    "file_name": "nba_alpaca.json",
    "format": "alpaca"
  },
```
Cela permet au programme de retrouver les données précédement
converti au format alpaca.

## Lancement du training
Quand les fichiers sont à leur place, on peut lancer le training.
Pour cela, il faut dans le repertoire LLamaFactory lancer la
commande : ``llamafactory-cli train config_training.yaml``. 
Cela va lancer l'entrainement sur les données spécifiées.
De plus le LoRA sera directement push sur le repo Hugging Face
grâce à la config.