# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from htc.models.common.HTCLightning import HTCLightning
from htc.models.data.DataSpecification import DataSpecification
from htc.utils.Config import Config
from htc_projects.sepsis_icu.models.DatasetMeta import DatasetMeta
from htc_projects.sepsis_icu.utils import config_meta_selection


def classifier_results(
    classifier_type, config: Config, test_results: bool = False, **kwargs
) -> dict[str, pd.DataFrame | np.ndarray]:
    """
    Run a classic classifier on the metadata.

    Args:
        classifier_type: The sklearn type of classifier to use.
        config: The training configuration object.
        test_results: If True, compute the results on the test set as well by ensembling the predictions from the different folds.
        **kwargs: Additional keyword arguments to pass to the classifier.

    Returns:
        A dictionary containing the results of the classifier:
        - df: A table with the predictions for each fold.
        - attributions: A matrix with the feature importances for each fold [n_folds, n_metadata].
    """
    spec = DataSpecification.from_config(config)
    if test_results:
        spec.activate_test_set()
        test_scores = []
        test_image_labels = []

    rows = {
        "fold_name": [],
        "subject_name": [],
        "image_name": [],
        "timestamp": [],
        "predictions": [],
        "image_labels": [],
    }
    attributions = []

    LightningClass = HTCLightning.class_from_config(config)
    for fold_name, splits in spec:
        dataset_train = LightningClass.dataset(paths=splits["train"], train=False, config=config)
        samples_train = dataset_train[:]
        scaler = StandardScaler()

        X_train = []
        if "features" in samples_train:
            X_train.append(samples_train["features"].numpy())
        if "meta" in samples_train:
            X_train.append(samples_train["meta"].numpy())
        X_train = np.concatenate(X_train, axis=1)

        X_train = scaler.fit_transform(X_train)
        y_train = samples_train["image_labels"].numpy()

        estimator = classifier_type(random_state=0, **kwargs)
        estimator.fit(X_train, y_train)

        dataset_val = LightningClass.dataset(paths=splits["val"], train=False, config=config)
        samples_val = dataset_val[:]

        X_val = []
        if "features" in samples_val:
            X_val.append(samples_val["features"].numpy())
        if "meta" in samples_val:
            X_val.append(samples_val["meta"].numpy())
        X_val = np.concatenate(X_val, axis=1)

        X_val = scaler.transform(X_val)
        y_val = samples_val["image_labels"].numpy()

        score_val = estimator.predict_proba(X_val)

        if test_results:
            dataset_test = LightningClass.dataset(paths=spec.paths("test"), train=False, config=config)
            samples_test = dataset_test[:]

            X_test = []
            if "features" in samples_test:
                X_test.append(samples_test["features"].numpy())
            if "meta" in samples_test:
                X_test.append(samples_test["meta"].numpy())
            X_test = np.concatenate(X_test, axis=1)

            X_test = scaler.transform(X_test)
            y_test = samples_test["image_labels"].numpy()

            test_scores.append(estimator.predict_proba(X_test))
            test_image_labels.append(y_test)

        rows["fold_name"] += [fold_name] * len(splits["val"])
        rows["subject_name"] += [p.subject_name for p in splits["val"]]
        rows["image_name"] += [p.image_name() for p in splits["val"]]
        rows["timestamp"] += [p.timestamp for p in splits["val"]]
        rows["predictions"] += score_val.tolist()
        rows["image_labels"] += list(y_val)

        if isinstance(estimator, RandomForestClassifier):
            attributions.append(estimator.feature_importances_)

    df = pd.DataFrame(rows)
    if isinstance(estimator, RandomForestClassifier):
        attributions = np.stack(attributions)

    if test_results:
        test_rows = {}
        test_scores = np.stack(test_scores)
        test_rows["predictions"] = np.mean(test_scores, axis=0).tolist()
        test_image_labels = np.stack(test_image_labels)
        for i in np.arange(1, test_image_labels.shape[0]):
            assert all(test_image_labels[0, :] == test_image_labels[i, :]), (
                f"Image labels are not the same for fold {i} as for fold 0"
            )
        test_rows["image_labels"] = list(test_image_labels[0, :])
        test_rows["subject_name"] = [p.subject_name for p in spec.paths("test")]
        test_rows["image_name"] = [p.image_name() for p in spec.paths("test")]
        test_rows["timestamp"] = [p.timestamp for p in spec.paths("test")]
        df_test = pd.DataFrame(test_rows)

        return {
            "df": df,
            "df_test": df_test,
            "attributions": attributions,
        }
    else:
        return {
            "df": df,
            "attributions": attributions,
        }


def recursive_feature_elimination(config: Config, rows: dict) -> list[Config, dict, str]:
    """
    Perform recursive feature elimination using the Random Forest Classifier. In opposite to RFE and RFECV, this function aggregates the feature importances over all folds before removing the least important feature.

    Args:
        config: The training configuration object.
        rows: A dictionary to store the results.

    Returns: A tuple containing the following:
        - The updated configuration object with the removed least important feature.
        - The updated dictionary with the results for the respective classifier.
        - The name of the removed feature.
    """
    attributions = []
    named_features = [c["name"] for c in config["input/meta/attributes"]]
    assert len(named_features) == len(set(named_features)), f"Duplicate feature names in {named_features}"
    current_n_features = len(named_features)
    spec = DataSpecification.from_config(config)

    for fold_name, splits in spec:
        dataset_train = DatasetMeta(paths=splits["train"], train=False, config=config)
        scaler = StandardScaler()
        X_train = dataset_train[:]["meta"].numpy()
        X_train = scaler.fit_transform(X_train)
        y_train = dataset_train[:]["image_labels"].numpy()

        estimator = RandomForestClassifier(random_state=0)
        estimator.fit(X_train, y_train)
        attributions.append(estimator.feature_importances_)

        dataset_val = DatasetMeta(paths=splits["val"], train=False, config=config)
        X_val = dataset_val[:]["meta"].numpy()
        X_val = scaler.transform(X_val)
        y_val = dataset_val[:]["image_labels"].numpy()
        score_val = estimator.predict_proba(X_val)

        rows["n_features"] += [current_n_features] * len(splits["val"])
        rows["used_features"] += [named_features] * len(splits["val"])
        rows["fold_name"] += [fold_name] * len(splits["val"])
        rows["subject_name"] += [p.subject_name for p in splits["val"]]
        rows["image_name"] += [p.image_name() for p in splits["val"]]
        rows["timestamp"] += [p.timestamp for p in splits["val"]]
        rows["predictions"] += score_val.tolist()
        rows["image_labels"] += list(y_val)

    attributions = np.stack(attributions)
    attributions = np.mean(attributions, axis=0)
    new_named_features = named_features.copy()
    removed_feature = new_named_features[np.argmin(attributions)]
    new_named_features.remove(removed_feature)
    assert len(new_named_features) == current_n_features - 1
    config = config_meta_selection(config, attribute_names=new_named_features)

    return config, rows, removed_feature


def feature_selection(config: Config) -> dict[str, pd.DataFrame | np.ndarray]:
    """
    Perform feature selection using Recursive Feature Elimination (RFE) with the Random Forest Classifier.

    Args:
        config: The training configuration object.

    Returns: A dictionary containing the following:
        - df: A table with the predictions for each selection of features (`top_n_features` = predictions of a classifier using the top n most important features).
        - rankings: Matrix with the feature rankings for each fold [n_folds, n_metadata].
    """
    config = copy.copy(config)

    n_features = len(config["input/meta/attributes"])
    rows = {}
    rows["n_features"] = []
    rows["used_features"] = []
    rows["fold_name"] = []
    rows["subject_name"] = []
    rows["image_name"] = []
    rows["timestamp"] = []
    rows["predictions"] = []
    rows["image_labels"] = []
    overall_ranking = []

    for i in np.arange(n_features):
        config, rows, removed_feature = recursive_feature_elimination(config, rows)
        overall_ranking.append(removed_feature)

    df = pd.DataFrame(rows)
    overall_ranking.reverse()  # change order from most important to least important

    return {
        "df": df,
        "overall_ranking": overall_ranking,
    }
