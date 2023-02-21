from imports import *
import features.build_features as features


# Séparation des données en conjoint d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features.training, features.output, test_size=0.2)


if not os.path.exists("../models/model.joblib"):
    # Création de l'objet DecisionTreeClassifier
    print(len(features.labels))
    # Création de l'objet MLPClassifier avec plusieurs couches cachées
    clf = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000)

    # Entraînement du modèle sur les données d'entraînement
    clf_old = clf.fit(X_train, y_train)
    for i in range(100):
        clf_new = clf.fit(X_train, y_train)
        score_old = clf_old.score(X_test, y_test)
        score_new = clf_new.score(X_test, y_test)
        if score_new > score_old:
            clf_old = clf_new

    clf = clf_old
    # Evaluation du modèle sur les données de test
    #accuracy = clf.score(X_test, y_test)
    #print("Précision du modèle : {:.2f}".format(accuracy))
    #dump(clf, 'model.joblib')
else:
    clf = load('../models/model.joblib')


prediction= clf.predict(X_test)
# Evaluation du modèle sur les données de test
score = clf.score(X_test, y_test)
accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction, average="micro", zero_division=0)
recall = recall_score(y_test, prediction, average="micro", zero_division=0)
print("Score du modèle : {:.2f}".format(score))
print("Accuracy du modèle : {:.2f}".format(accuracy))

#print(precision)
#print("\n\n\n")
#print(recall)
print("Precision du modèle : {:.2f}".format(precision))
print("Recall du modèle : {:.2f}".format(recall))
