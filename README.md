# Medibot

Projet de chatbot médical qui peut prescrire les premier soins a donner en cas d'accident ou d'urgence médicale grace a une base de donnée en ".json".

<h2>Technologies Utilisé:</h2>
Pour realiser se Chat-bot j'ai utiliser le language de programmation Python avec la bibliothéque de machine learning SktLearn, pour l'entrainement du chat j'ai testé deux algorithme: l'arbre de décision et le deep-learning.

Pour la préparation des donnée on a utilisée un fichier ".json" pour les recolter sous forme de tag et les transformer en sac de mots grace a la bibliothéque NLTK de python qui permet de traiter plus facilement et efficacement le language humain

Apres le traitement et l'entrainement des donnée on recupere l'input de l'utilisateur afin de le transformer en sac de mots et predire la réponse la plus pertinente.

**A savoir:** j'ai sauvegardé les meilleurs résultats de l'entrainement dans un modéle afin d'eviter de toujours devoir le reentrainer a chaque execution et par consequent accelerer l'execution du chat-bot.

<h2>Résultats Obtenu</h2>
	-On a obtenus un taux de précision de 95% pour notre Chatbot
	-On a choisi l’arbre de décision comme modèle finale car le Deep Learning est plus adapté sur TenserFlow méme-ci dans la realité pour faire un chat bot beaucoup plus complexe, il faut utiliser le Deep en plus dans notre cas le fait de transformer nos données en sac de mots fait que le descision tree deviens le plus pertinent.

<h2>Installation du projet</h2>
Il faut vous rendre dans le dossier **/src** c'est la ou il y a le fichier executable.
1/- Installer NLTK grace a la commande **pip3 install nltk** 
2/- Installer Sklearn grace a la commande **pip3 install sklearn**
3/- Installer Numpy grace a la commande **pip3 install numpy**
4/- Maintenant vous pouvez executer le programme en etant toujours dans le dossier src/, avec la commande **python3 chat-bot-decision-tree.py**

<h3>PS:</h3> avant tous ca, assurez vous d'avoir deja installer pip3 grace a la commande: **sudo apt-get update && sudo apt install python3-pip**

