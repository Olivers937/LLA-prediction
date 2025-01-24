from pathlib import Path
from typing import List

import cv2
import numpy as np
import typer
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops

app = typer.Typer()

def extract_features(image: np.ndarray) -> np.ndarray:
    features = []

    # 1. Caractéristiques morphologiques (spécifiques à la LLA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features.extend(_extract_morphological_features(gray))

    # 2. Caractéristiques de couleur (importantes pour la LLA)
    features.extend(_extract_color_features(image))

    # 3. Caractéristiques de texture
    features.extend(_extract_texture_features(gray))

    return np.array(features, dtype=np.float32)

def _extract_morphological_features(gray: np.ndarray) -> List[float]:
    features = []

    # Segmentation des cellules
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Caractéristiques de forme des cellules
        areas = [cv2.contourArea(cnt) for cnt in contours]
        perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
        circularities = [(4 * np.pi * area) / (peri ** 2) if peri > 0 else 0
                         for area, peri in zip(areas, perimeters)]

        features.extend([
            np.mean(areas),  # Taille moyenne des cellules
            np.std(areas),  # Variation de taille
            np.mean(circularities),  # Circularité moyenne
            np.std(circularities),  # Variation de forme
            len(contours)  # Nombre de cellules
        ])
    else:
        features.extend([0] * 5)

    return features


def _extract_color_features(image: np.ndarray) -> List[float]:
    features = []

    # Conversion en espace LAB (plus proche de la perception humaine)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    for channel in cv2.split(lab):
        features.extend([
            np.mean(channel),  # Moyenne
            np.std(channel),  # Écart-type
            np.percentile(channel, 25),  # Q1
            np.percentile(channel, 75),  # Q3
            skew(channel.ravel()),  # Asymétrie
            kurtosis(channel.ravel())  # Kurtosis
        ])

    return features


def _extract_texture_features(gray: np.ndarray) -> List[float]:
    features = []

    # GLCM amélioré
    glcm = graycomatrix(gray, [1], [0, 45, 90, 135],
                        levels=256, symmetric=True, normed=True)

    # Caractéristiques de Haralick
    contrast = graycoprops(glcm, 'contrast').ravel()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()

    features.extend([
        np.mean(contrast),
        np.mean(dissimilarity),
        np.mean(homogeneity),
        np.mean(correlation)
    ])

    return features


if __name__ == "__main__":
    app()
