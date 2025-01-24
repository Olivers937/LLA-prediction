binary-leukemia-detector.py est la version finale pour la LLA (bon la LMA, à cause du dataset)

ensemble-classifier.py permet de clzssfier les image de façon général. (Javais oubié de spécifier le contete, vu quej'avais changé de compte comme mes tokens etaient bolès)

ensemble-leukemia-detector.py Ça c'est pour tout les types de leucemie. (Ça fait partie des productions obbtenues faute de manque de contextualisation lorsque je switch de compte)

leukemia-detector.py ça c'est en utilisant les RandomForest, mais avec en plus les réseaux de neuronnes, donc c'est pas à utiliser. Mais il s'est inspirer de ça pour fair e celuis sans les NN (le suivant)

rf-leukemia-detector.py C'est RF sans NN. Il entre dans la conception du modée final (le premier model)

ml-pipeline.py permet de charger les images et de prédire de manière obtimisé. (j'ai oublié ce que ça utilisait)


L'architecture de la version finale: 

On ulise les modèle Lr, RF, ETC, et SVM poour prédire, et avec on passe les résultats à un autre model pour qu'il utilise la plus grande occurence des prédictions pour faire la prédiction finale.

