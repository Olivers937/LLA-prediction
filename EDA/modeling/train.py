import logging
import sys
from pathlib import Path
from typing import List, Any, Dict

import numpy as np
import pandas as pd
import typer
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm
from models.base_model_wrapper import BaseModelWrapper
from EDA.datamanagement import load_dataset
from EDA.config import Config as config
import time
sys.path.append(str(Path(__file__).resolve().parent.parent))

app = typer.Typer()


@app.command()
def main(model_choice : int = 4):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------

    logging.info("Chargement et traitement des données...")
    malign_dir = [config.malignant_pre_b_dir.absolute(), config.malignant_pro_b_dir.absolute(), config.malignant_early_pre_b_dir.absolute()]
    X, y = load_dataset([config.benign_dir.absolute()], malign_dir)
    # print(f"\nprint loaded dataset\nX = {X.shape} y = {y.shape}")
    np.savetxt("test.txt", y)

    # Division des données
    tick = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )

    print(f"After train_test_split X_test = {len(X_test)}")

    # Entraînement parallèle des modèles

    # with ThreadPoolExecutor(max_workers=len(models)) as executor:
    #   futures = [executor.submit(model.fit, X_train, y_train)
    #            for model in models]
    #    for future in futures:
    #        tick = time.time()
    #        future.result()
    #        tock = time.time()
    #        print(f"time = {tock - tick}")
    models = _initialize_models(model_choice)
    for model in models:
        logging.info(f"start training {model.name}")
        tick = time.time()
        model.fit(X_train, y_train)
        tock = time.time()
        logging.info(f"end training {model.name}")
        print(f"time = {tock - tick} for model {model.name}")
        np.savetxt("test.txt", y_train)

    # X_test, y_test = _load_dataset(benign_training_dir, malign_training_dir)

    print(f"X_test= {X_test.shape} y_test = {y_test.shape}")

    result = _evaluate_ensemble(X_test, y_test, models)
    print(f"result = {result}")
    # _save_model()

    return result


def _initialize_models(model_choice: int ) -> list[Any]:
    """Initialise les modèles de l'ensemble"""
    if model_choice == 4:
        models = [
            BaseModelWrapper(
                RandomForestClassifier(random_state=config.random_state),
                config.rf_params,
                "Random Forest",
            ),
            BaseModelWrapper(
                SVC(probability=True, random_state=config.random_state),
                config.svm_params,
                "SVM",
            ),
            BaseModelWrapper(
                ExtraTreesClassifier(random_state=config.random_state),
                config.etc_params,
                "Extra Trees",
            ),
            BaseModelWrapper(
                LogisticRegression(random_state=config.random_state),
                config.lr_params,
                "Logistic Regression",
            )
        ]
        return models
    if model_choice == 0:
        return [
            BaseModelWrapper(RandomForestClassifier(
                random_state=config.random_state),
                config.rf_params,
                "Random Forest",),
        ]
    elif model_choice == 1:
        return [
            BaseModelWrapper(
                SVC(probability=True, random_state=config.random_state),
                config.svm_params,
                "SVM",
            )
        ]
    elif model_choice == 2:
        return [
            BaseModelWrapper(
                ExtraTreesClassifier(random_state=config.random_state),
                config.etc_params,
                "Extra Trees",
            )
        ]
    else :
        return [
        BaseModelWrapper(
            LogisticRegression(random_state=config.random_state),
            config.lr_params,
            "Logistic Regression",
        )
    ]
    
def _evaluate_ensemble(X_test: np.ndarray, y_test: np.ndarray, models:[]) -> Dict[str, Any]:
    """Évalue les performances de l'ensemble"""
    # Prédictions individuelles
    individual_preds = {model.name: model.predict(X_test) for model in models}
    individual_probas = {model.name: model.predict_proba(X_test)[:, 1]
                      for model in models}
    result = 0
    for model in models:
        plt_test_result(X_test, y_test, individual_preds[model.name], model.name)
        print(model.to_string())

    for t in range(len(y_test)):
        if individual_preds[models[0].name][t] == individual_preds[models[1].name][t] == individual_preds[models[2].name][t] == individual_preds[models[3].name][t] :
            result += 1
    data =  {
        "y_test": y_test,
        models[0].name: individual_preds[models[0].name],
        models[1].name: individual_preds[models[1].name],
        models[2].name: individual_preds[models[2].name],
        models[3].name: individual_preds[models[3].name],
        models[0].name+"_proba": individual_probas[models[0].name],
        models[1].name+"_proba": individual_probas[models[1].name],
        models[2].name+"_proba": individual_probas[models[2].name],
        models[3].name+"_proba": individual_probas[models[3].name],
    }
    df = pd.DataFrame(data)
    df.to_csv("results.csv")

    print(f"result = {result}")
    print(f"y_test = {len(y_test)}")


def plt_test_result(X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, name: str):

    plt.figure(figsize=(10, 6))

    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', cmap='viridis', label='Réel', alpha=0.7)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x', cmap='coolwarm', label='Prédit', alpha=0.7)

    plt.title(f"Comparaison entre étiquettes réelles et prédites : {name}", fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.show()






if __name__ == "__main__":
    print(f"malignant_pro_b_dir = {config.malignant_pro_b_dir.absolute()}")
    args = sys.argv[1:]
    if len(args) > 0 :
        if args[0].isdigit():
            app(args[0])
    app()
