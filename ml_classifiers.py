import os
import graphviz
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint as pp
from joblib import dump
from typing import Union
from datetime import datetime

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    train_test_split,
    KFold,
    RepeatedKFold,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class Classifiers(object):
    def __init__(self, feature_scaler: str = None, random_state: int = 123):
        """
        데이터셋에 적용할 Feature scaler를 지정한다.
        (지정하지 않을 경우 원본 데이터를 그대로 예측에 사용)

        :param feature_scaler: str, default=None
            Transform features by scaling each feature to a given range.
            - `StandardScaler`
            - `MinMaxScaler`
            - `RobustScaler`
            - `MaxAbsScaler`
            - `Normalizer`

        """

        super(Classifiers, self).__init__()
        self.random_state = random_state
        self.feature_scaler = feature_scaler
        self.label_count = None
        self.n_splits = None
        self.scoring = None
        self.feature_names = None
        self.df_hyperparams_search_result = None
        self.best_estimators = None
        self.n_repeats = None
        self.df_cv_result = None

    def _get_feature_scaler(self):
        if self.feature_scaler == "StandardScaler":
            from sklearn.preprocessing import StandardScaler

            return StandardScaler()

        elif self.feature_scaler == "MinMaxScaler":
            from sklearn.preprocessing import MinMaxScaler

            return MinMaxScaler()

        elif self.feature_scaler == "RobustScaler":
            from sklearn.preprocessing import RobustScaler

            return RobustScaler()

        elif self.feature_scaler == "MaxAbsScaler":
            from sklearn.preprocessing import MaxAbsScaler

            return MaxAbsScaler()

        elif self.feature_scaler == "Normalizer":
            from sklearn.preprocessing import Normalizer

            return Normalizer()

        else:
            print(
                "No scaler conditions are satisfied. "
                "`feature_scaler` could be one of "
                "`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler` or `Normalizer`. "
                "If you don't want use it, apply other scaler first. "
            )
            return None

    def _transform_feature_scale(self, X, y) -> tuple:
        """
        데이터셋 단위를 정규화하여 반환한다.

        :param X: Features to fit and train model.
        :param y: A target label to predict.
        """

        self.label_count = np.unique(y).size

        if self._get_feature_scaler() is None:
            return X, y
        else:
            scaler = self._get_feature_scaler()
            scaler.fit(X)
            X_scaled = scaler.transform(X)
            return X_scaled, y

    def split_data_into_train_test(
        self, X, y, test_size: float = None, shuffle: bool = True
    ) -> tuple:
        """
        모델링에 사용할 훈련, 검증 데이터를 분리한다.

        :param X: Features to fit and train model.
        :param y: A target label to predict.

        :param test_size : float or int, default=None
            If floated, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size.

        :param shuffle: bool, default=True
            Whether to shuffle the data before splitting. If shuffle=False
            then stratify must be None.
        """

        X, y = self._transform_feature_scale(X=X, y=y)
        self.label_count = np.unique(y).size
        return train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=shuffle,
            random_state=self.random_state,
            stratify=y,
        )

    def _get_kfold_function(self, kfold: str, n_splits: int = 5, n_repeats: int = 10):
        """
        교차 검증에 적용할 방법을 선택한다.

        :param kfold: str
            K-Folds cross-validator.
            (e.g, "KFold", "StratifiedKFold", "RepeatedKFold" or "RepeatedStratifiedKFold")

        :param n_splits: int, default=5
            Number of folds. Must be at least 2.

        :param n_repeats, default=10
            Number of times cross-validator needs to be repeated.
        """

        if kfold == "KFold":
            return KFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
        elif kfold == "StratifiedKFold":
            return StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
        elif kfold == "RepeatedKFold":
            return RepeatedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state
            )
        elif kfold == "RepeatedStratifiedKFold":
            return RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state
            )
        else:
            raise ValueError(
                "kfold should be ond of "
                "`KFold`, `StratifiedKFold`, `RepeatedKFold` or `RepeatedStratifiedKFold`."
            )

    def _get_classifier_models(self) -> dict:
        """
        예측에 사용할 알고리즘 사전 정의
        """

        return {
            "MLPClassifier": MLPClassifier,
            "KNeighborsClassifier": KNeighborsClassifier,
            "SVC": SVC,
            "GaussianProcessClassifier": GaussianProcessClassifier,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "ExtraTreesClassifier": ExtraTreesClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "AdaBoostClassifier": AdaBoostClassifier,
            "GaussianNB": GaussianNB,
            "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "LogisticRegression": LogisticRegression,
            "XGBClassifier": XGBClassifier,
            "LGBMClassifier": LGBMClassifier,
        }

    def run_cross_validation(
        self,
        X,
        y,
        estimators: Union[str, list] = "all",
        kfold: str = None,
        n_splits: int = None,
        n_repeats: int = None,
        scoring: Union[str, list] = None,
        estimator_params: dict = None,
    ):
        """
        교차 검증 결과를 반환한다.

        :param X: Features to fit and train model.
        :param y: A target label to predict.

        :param estimators: str or list
            The estimator list to use to fit the data.
            (e.g, "all" or ["DecisionTreeClassifier", "RandomForestClassifier"]..)

        :param kfold: str, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for kfold are:
                - "KFold"
                - "StratifiedKFold"
                - "RepeatedKFold"
                - "RepeatedStratifiedKFold"

        :param n_splits: int, default=5
            Number of folds. Must be at least 2.

        :param n_repeats, default=10
            Number of times cross-validator needs to be repeated.

        :param scoring: str, callable
            Strategy to evaluate the performance of the cross-validated model on the test set.
            Example for scoring are:
                - "accuracy"
                - "recall"
                - "precision"
                - "f1"
                - "roc_auc"
                - make_scorer(precision, average="micro")
                - ...

        :param estimator_params: dict
            Parameters to pass to the fit method of the estimator.
        """

        if estimators == "all":
            classifiers = self._get_classifier_models()
        else:
            estimators_intersection = list(
                set(estimators) & set(self._get_classifier_models().keys())
            )
            classifiers = {}
            for label in estimators_intersection:
                classifiers.update({label: self._get_classifier_models()[label]})
        classifiers = dict(sorted(classifiers.items()))

        if estimator_params is not None:
            if len(list(set(estimator_params.keys()) & set(classifiers.keys()))) == 0:
                raise ValueError(
                    " If you use `estimator_params`, "
                    "you must also include estimator in the list of `estimators`. "
                )

        X, y = self._transform_feature_scale(X=X, y=y)
        self.df_cv_result = pd.DataFrame()
        self.label_count = np.unique(y).size

        for estimator_name, estimator in classifiers.items():
            print(
                f" \n---------------------- {estimator_name} ---------------------- "
            )
            fit_start_time = datetime.now()

            if estimator_params is not None:
                if estimator_name in estimator_params.keys():
                    estimator_params = estimator_params[estimator_name]
                    if "fit_params" in estimator_params.keys():
                        fit_params = estimator_params["fit_params"]
                        estimator_params.pop("fit_params")
                    else:
                        fit_params = None
                else:
                    estimator_params = None
                    fit_params = None
            else:
                estimator_params = None
                fit_params = None

            print(f" >>> estimator_params: {estimator_params}")
            print(f" >>> fit_params: {fit_params}")
            if self.label_count >= 2:
                cv_result_dict = cross_validate(
                    estimator=estimator()
                    if estimator_params is None
                    else estimator(**estimator_params),
                    fit_params=fit_params,
                    X=X,
                    y=y,
                    cv=self._get_kfold_function(
                        kfold=kfold, n_splits=n_splits, n_repeats=n_repeats
                    ),
                    scoring=scoring,
                    n_jobs=-1,
                )
                df_cv = pd.DataFrame(data=cv_result_dict)

                if isinstance(scoring, list):
                    value_vars = [f"test_{s}" for s in scoring]
                    df_cv = df_cv.melt(
                        id_vars=["fit_time", "score_time"], value_vars=value_vars
                    )
                    df_cv = df_cv.rename(
                        columns={"variable": "scoring", "value": "test_score"}
                    )
                    df_cv["scoring"] = df_cv["scoring"].apply(
                        lambda x: x.replace("test_", "")
                    )
                else:
                    df_cv["scoring"] = scoring
                df_cv["estimator_name"] = estimator_name
                self.df_cv_result = pd.concat(
                    [self.df_cv_result, df_cv], ignore_index=True
                )

            else:
                raise ValueError(" Single class can not create confusion matrix. ")
            print(
                f" >>> Finished. (elapsed_time: {(datetime.now() - fit_start_time).seconds}s) "
            )

    def show_cross_validation_result(self):
        """
        교차 검증 결과를 시각화한다.
        """

        df_cv_result_agg = (
            self.df_cv_result.groupby(["estimator_name", "scoring"])
            .agg(
                mean_test_score=("test_score", "mean"),
                mean_fit_time=("fit_time", "mean"),
            )
            .reset_index()
        )

        df = self.df_cv_result.merge(
            right=df_cv_result_agg, on=["estimator_name", "scoring"], how="inner"
        ).reset_index(drop=True)

        df = df.sort_values(
            by=["scoring", "mean_test_score", "estimator_name"],
            ascending=[True, False, True],
            ignore_index=True,
        )

        sns.set(rc={"figure.figsize": (8, 12), "figure.dpi": 120})
        for scoring in df["scoring"].unique():
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.4)
            sns.boxplot(
                data=df[df["scoring"] == scoring],
                x="test_score",
                y="estimator_name",
                ax=ax1,
                showmeans=True,
                meanprops={
                    "marker": "v",
                    "markerfacecolor": "white",
                    "markeredgecolor": "white",
                },
            )
            sns.boxplot(
                data=df[df["scoring"] == scoring],
                x="fit_time",
                y="estimator_name",
                ax=ax2,
                showmeans=True,
                meanprops={
                    "marker": "v",
                    "markerfacecolor": "white",
                    "markeredgecolor": "white",
                },
            )
            for ax in [ax1, ax2]:
                ax.tick_params(axis="x", labelsize=14)
                ax.tick_params(axis="y", labelsize=14)
                ax.set_xlabel(xlabel=None)
                ax.set_ylabel(ylabel=None)
            ax1.set_title(str(scoring).upper(), fontsize=14)
            ax2.set_title("fit time (sec)", fontsize=14)

            df_tmp = df.loc[
                df["scoring"] == scoring,
                ["estimator_name", "mean_test_score", "mean_fit_time"],
            ].drop_duplicates(ignore_index=True)

            for index, row in df_tmp.iterrows():
                ax1.text(
                    x=row.mean_test_score,
                    y=index,
                    s=round(row.mean_test_score, 3),
                    color="black",
                    ha="center",
                    fontsize=13,
                )
                ax2.text(
                    x=row.mean_fit_time,
                    y=index,
                    s=round(row.mean_fit_time, 3),
                    color="black",
                    ha="center",
                    fontsize=13,
                )
            plt.show()

    def generate_params_grid(self, hyperparams_space: dict) -> dict:
        """
        하이퍼파라미터 튜닝을 시도할 알고리즘과 해당 파라미터 Grid를 반환한다.

        :param hyperparams_space: dictionary
        """

        if "XGBClassifier" in hyperparams_space.keys():
            if self.label_count > 2:
                hyperparams_space["XGBClassifier"]["objective"] = ["multi:softprob"]
            else:
                hyperparams_space["XGBClassifier"]["objective"] = ["binary:logistic"]

        if "LGBMClassifier" in hyperparams_space.keys():
            if self.label_count > 2:
                hyperparams_space["LGBMClassifier"]["objective"] = ["multiclass"]
            else:
                hyperparams_space["LGBMClassifier"]["objective"] = ["binary"]
        return hyperparams_space

    def search_hyperparameter(
        self,
        X,
        y,
        hyperparams_space: dict,
        feature_names: list = None,
        search_method: str = "random",
        test_size: float = 0.25,
        shuffle: bool = True,
        kfold: str = None,
        n_splits: int = None,
        n_repeats: int = None,
        scoring: str = None,
        n_iter: int = 10,
        factor: int = 3,
        filepath: str = "model_saved",
        verbose: int = -1,
    ):
        """
        하이퍼파라미터 탐색 결과를 반환한다.

        :param X: Features to fit and train model.
        :param y: A target label to predict.

        :param feature_names: list
            Name of features.

        :param hyperparams_space: dictionary
            model and hyperparams to use

        :param search_method: str, default="ramdom"
            Determines the automated hyperparameter selection strategy.
            Possible inputs for kfold are:
                - "grid"
                - "random"
                - "grid_halving"
                - "random_halving"

        :param test_size: test sample size
        :param shuffle: to use data shuffling

        :param kfold: str, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for kfold are:
                - "KFold"
                - "StratifiedKFold"
                - "RepeatedKFold"
                - "RepeatedStratifiedKFold"

        :param n_splits: int, default=5
            Number of folds. Must be at least 2.

        :param n_repeats, default=10
            Number of times cross-validator needs to be repeated.

        :param scoring: str, callable
            Strategy to evaluate the performance of the cross-validated model on the test set.
            Example for scoring are:
                - "accuracy"
                - "recall"
                - "precision"
                - "f1"
                - "roc_auc"
                - make_scorer(precision, average="micro")

        :param n_iter, default=10
            Number of parameter settings that are produced.

        :param factor: int or float, default=3
            The 'halving' parameter, which determines the proportion of candidates
            that are selected for each subsequent iteration. For example,
            ``factor=3`` means that only one third of the candidates are selected.

        :param filepath: str, default="model_saved"
            The filepath to save the best estimator.
        """

        self.feature_names = feature_names
        self.df_hyperparams_search_result = pd.DataFrame()
        self.best_estimators = dict()
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.scoring = scoring

        X_train, X_test, y_train, y_test = self.split_data_into_train_test(
            X=X, y=y, test_size=test_size, shuffle=shuffle
        )

        classifier_all = self._get_classifier_models().keys()
        params_grid = self.generate_params_grid(hyperparams_space=hyperparams_space)
        classifier_to_use = params_grid.keys()

        comm_classifiers = classifier_all & classifier_to_use
        classifiers = {}
        for label in comm_classifiers:
            classifiers.update({label: self._get_classifier_models()[label]})
        classifiers = dict(sorted(classifiers.items()))

        for estimator_name, estimator in classifiers.items():
            print(
                f" \n---------------------- {estimator_name} ---------------------- "
            )
            fit_start_time = datetime.now()

            param_grid = params_grid[estimator_name]
            if "fit_params" in param_grid.keys():
                fit_params = param_grid["fit_params"]
                param_grid.pop("fit_params")
            else:
                fit_params = None

            print(" >>> Hyperparameter Space: ")
            pp.pprint(param_grid)
            print(f" >>> fit_params: {fit_params}")

            if search_method == "grid":
                model = GridSearchCV(
                    estimator=estimator(),
                    param_grid=param_grid,
                    cv=self._get_kfold_function(
                        kfold=kfold, n_splits=n_splits, n_repeats=n_repeats
                    ),
                    scoring=scoring,
                    n_jobs=-1,
                    return_train_score=True,
                    verbose=verbose,
                )

            elif search_method == "random":
                model = RandomizedSearchCV(
                    estimator=estimator(),
                    param_distributions=param_grid,
                    cv=self._get_kfold_function(
                        kfold=kfold, n_splits=n_splits, n_repeats=n_repeats
                    ),
                    scoring=scoring,
                    n_iter=n_iter,
                    n_jobs=-1,
                    return_train_score=True,
                    verbose=verbose,
                )

            elif search_method == "grid_halving":
                model = HalvingGridSearchCV(
                    estimator=estimator(),
                    param_grid=param_grid,
                    cv=self._get_kfold_function(
                        kfold=kfold, n_splits=n_splits, n_repeats=n_repeats
                    ),
                    scoring=scoring,
                    n_jobs=-1,
                    return_train_score=True,
                    min_resources="exhaust",
                    factor=factor,
                    verbose=verbose,
                )

            elif search_method == "random_halving":
                model = HalvingRandomSearchCV(
                    estimator=estimator(),
                    param_distributions=param_grid,
                    cv=self._get_kfold_function(
                        kfold=kfold, n_splits=n_splits, n_repeats=n_repeats
                    ),
                    scoring=scoring,
                    n_jobs=-1,
                    return_train_score=True,
                    n_candidates="exhaust",
                    factor=factor,
                    verbose=verbose,
                )

            else:
                raise ValueError(
                    "`search_method` should be ond of 'grid', 'random', 'grid_halving' or 'random_halving'."
                )

            if (estimator_name == "XGBClassifier") or (
                estimator_name == "LGBMClassifier"
            ):
                if fit_params is None:
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
                else:
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_test, y_test)],
                        **fit_params,
                    )
            else:
                if fit_params is None:
                    model.fit(X_train, y_train)
                else:
                    model.fit(
                        X_train,
                        y_train,
                        **fit_params,
                    )

            print(" >>> Best Parameters:")
            pp.pprint(model.best_params_)
            print(
                f" >>> Finished. (elapsed_time: {(datetime.now() - fit_start_time).seconds}s) "
            )
            self.best_estimators.update({estimator_name: model.best_estimator_})
            os.makedirs(filepath, exist_ok=True)
            dump(model.best_estimator_, f"{filepath}/{estimator_name}.joblib")

            cv_result_tmp = pd.DataFrame(data=model.cv_results_)
            cv_result_tmp["estimator_name"] = estimator_name
            self.df_hyperparams_search_result = pd.concat(
                [self.df_hyperparams_search_result, cv_result_tmp], ignore_index=True
            )

    def show_hyperparameter_search_result(self):
        """
        하이퍼파라미터 탐색 결과를 시각화한다.
        """

        df = (
            self.df_hyperparams_search_result[
                self.df_hyperparams_search_result.rank_test_score == 1
            ]
            .groupby("estimator_name")
            .head(1)
        )

        df = df.sort_values(by="mean_test_score", ascending=False, ignore_index=True)

        # 모델 별 performance metric score 생성
        df_score = pd.DataFrame()

        if self.n_repeats is None:
            N = self.n_splits  # score 개수: k-fold의 k
        else:
            N = self.n_splits * self.n_repeats  # score 개수: (k-fold의 k) x (cv 반복 횟수)

        for i in range(N):
            df_score = pd.concat(
                [
                    df_score,
                    df[[f"split{i}_test_score", "estimator_name"]].rename(
                        columns={f"split{i}_test_score": "test_score"}
                    ),
                ],
                ignore_index=True,
            )

        sns.set(rc={"figure.figsize": (8, 12), "figure.dpi": 120})
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.4)
        sns.boxplot(
            data=df_score,
            x="test_score",
            y="estimator_name",
            ax=ax1,
            showmeans=True,
            meanprops={
                "marker": "v",
                "markerfacecolor": "white",
                "markeredgecolor": "white",
            },
        )
        sns.barplot(
            data=df,
            x="mean_fit_time",
            y="estimator_name",
            ax=ax2,
        )
        for ax in [ax1, ax2]:
            ax.tick_params(axis="x", labelsize=14)
            ax.tick_params(axis="y", labelsize=14)
            ax.set_xlabel(xlabel=None)
            ax.set_ylabel(ylabel=None)
        ax1.set_title(str(self.scoring).upper(), fontsize=14)
        ax2.set_title("mean fit time (sec)", fontsize=14)

        for index, row in df.iterrows():
            ax1.text(
                x=row.mean_test_score,
                y=index,
                s=round(row.mean_test_score, 3),
                color="black",
                ha="center",
                fontsize=13,
            )
            ax2.text(
                x=row.mean_fit_time,
                y=index,
                s=round(row.mean_fit_time, 3),
                color="black",
                ha="center",
                fontsize=13,
            )

    def evaluate_model(self, y_test, pred, pred_proba=None) -> Union[dict, float]:
        """
        모델 성능 결과를 반환한다.

        :param y_test: 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) labels.

        :param pred: 1d array-like, or label indicator array / sparse matrix
            Predicted labels, as returned by a classifier.

        :param pred_proba: array-like of shape (n_samples,) or (n_samples, n_classes) Target scores.
        """

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        if self.label_count > 2:
            return accuracy_score(y_test, pred)
        else:
            accuracy = accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred)
            recall = recall_score(y_test, pred)
            f1 = f1_score(y_test, pred)
            roc_auc = roc_auc_score(y_test, pred_proba)
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            }

    def show_precision_recall_by_thresholds(self, X, y, estimators: str):
        """
        threshold에 따른 이진 분류 성능(정밀도, 재현율)을 시각화한다.
        """

        from sklearn.metrics import precision_recall_curve

        X_train, X_test, y_train, y_test = self.split_data_into_train_test(X=X, y=y)

        if estimators is None:
            best_estimators_ = self.best_estimators
        else:
            best_estimators_ = {}
            for estimator_name in estimators:
                best_estimators_.update(
                    {estimator_name: self.best_estimators[estimator_name]}
                )

        for estimator_name, estimator in best_estimators_.items():
            pred_proba_class1 = estimator.predict_proba(X_test)[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(
                y_test, pred_proba_class1
            )

            plt.figure(figsize=(12, 6))
            threshold_boundary = thresholds.shape[0]
            plt.plot(
                thresholds,
                precisions[0:threshold_boundary],
                linestyle="--",
                label="precision",
            )
            plt.plot(thresholds, recalls[0:threshold_boundary], label="recall")
            plt.title(f"Precision and Recall by Threshold ({estimator_name})")
            start, end = plt.xlim()
            plt.xticks(np.round(np.arange(start, end, 0.1), 2))
            plt.xlabel("Threshold")
            plt.ylabel("Precision and Recall")
            plt.legend()
            plt.grid()
            plt.show()

    def get_best_model_info(self):
        """
        하이퍼파라미터 탐색 결과로부터 최적의 모델의 파라미터를 반환한다.
            - Best 모델 선정 기준
            - 1) mean_test_score 높은순
            - 2) std_test_score 낮은순
            - 3) mean_fit_time 낮은순
        """

        cols = [
            "mean_test_score",
            "std_test_score",
            "mean_fit_time",
            "estimator_name",
            "params",
        ]
        return (
            self.df_hyperparams_search_result[cols]
            .sort_values(by=cols[:3], ascending=[False, True, True], ignore_index=True)
            .iloc[1, :]
            .to_dict()
        )

    def get_best_classifier(self, estimator_name: str = None):
        """
        예측에 사용할 최종 estimator를 반환한다.
        """

        if estimator_name is None:
            return self.best_estimators[self.get_best_model_info()["estimator_name"]]
        else:
            return self.best_estimators[estimator_name]

    def show_feature_importances(
        self, estimators: list = None, index: list = None, n_features: int = 10
    ):
        """
        훈련된 모델로 Feature Importance를 시각화한다.
        (훈련 데이터로 생성된 중요도이므로 이것에 너무 의존하지 않을 필요가 있음.)
        """

        feature_importances = {}
        if estimators is None:
            for estimator_name, estimator in self.best_estimators.items():
                try:  # tree 기반에 한해서만 feature_importances가 나옴
                    fi = (
                        estimator.feature_importances_
                        / estimator.feature_importances_.sum()
                    )
                    feature_importances.update({estimator_name: fi[:n_features]})
                except:
                    continue
        else:
            for estimator_name in estimators:
                try:
                    fi = (
                        self.best_estimators[estimator_name].feature_importances_
                        / self.best_estimators[
                            estimator_name
                        ].feature_importances_.sum()
                    )
                    feature_importances.update({estimator_name: fi[:n_features]})
                except:
                    continue

        sns.set(rc={"figure.figsize": (8, 12), "figure.dpi": 120})
        if index is None:
            df_tmp = pd.DataFrame(
                data=feature_importances, index=self.feature_names[:n_features]
            )
            for i in range(df_tmp.columns.size):
                df_tmp.iloc[:, i].sort_values().plot(
                    kind="barh",
                    title=f"Feature Importances ({df_tmp.columns[i]})",
                    figsize=(12, 8),
                    fontsize=14,
                )
                plt.yticks(fontsize=14)
                plt.show()
        else:
            df_tmp = pd.DataFrame(data=feature_importances, index=index)
            for i in range(df_tmp.columns.size):
                df_tmp.iloc[:, i].sort_values().plot(
                    kind="barh",
                    title=f"Feature Importances ({df_tmp.columns[i]})",
                    figsize=(12, 8),
                    fontsize=14,
                )
                plt.yticks(fontsize=14)
                plt.show()

    def show_permutation_importances(
        self, X, y, estimators: list = None, scoring: str = None
    ):
        """
        훈련된 모델로 각 Feature 여부에 따른 중요도를 시각화한다.
            1. 기존 test data에서 하나의 feature에 대해 row 순서를 무작위로 섞어서 새로운 test data를 생성
            2. 새로운 test data로 score를 측정
            3. 기존 test data에 의한 score 대비 새로운 test data에 의한 score 비교
                - score가 많이 감소한다면 해당 feature가 중요하다는 의미
                - score가 거의 그대로라면 해당 feature는 그닥 중요하진 않다는 의미
            4. score가 감소하지 않는 feature를 발견 시, 해당 feature를 제와하여 다시 튜닝 수행
                - 성능 개선 효과 기대 가능
                - feature 제거에 따른 추가 리소스 확보 가능

        :param data: pd.DataFrame
            Data on which permutation importance will be computed.

        :param target: str
            Target for supervised or `None` for unsupervised.

        :param estimators : list, default=None
            Estimators that has already been performed to fit

        :param scoring: str, list, tuple, default=None
            Scorer to use.
            If `scoring` represents a single score, one can use:
            - a single string (see :ref:`scoring_parameter`);

            If `scoring` represents multiple scores, one can use:
            - a list or tuple of unique strings;

            If None, the estimator's default scorer is used.
        """
        X_train, X_test, y_train, y_test = self.split_data_into_train_test(X=X, y=y)

        permutation_importances = {}
        if estimators is None:
            best_estimators_ = self.best_estimators
        else:
            best_estimators_ = {}
            for estimator_name in estimators:
                best_estimators_.update(
                    {estimator_name: self.best_estimators[estimator_name]}
                )

        for estimator_name, estimator in best_estimators_.items():
            scores = permutation_importance(
                estimator=self.best_estimators[estimator_name],
                X=X_test,
                y=y_test,
                scoring=scoring,
                n_repeats=10,
                n_jobs=-1,
                random_state=self.random_state,
            )
            permutation_importances.update({estimator_name: scores})

        sns.set(rc={"figure.figsize": (8, 12), "figure.dpi": 120})
        for estimator_name in permutation_importances.keys():
            if isinstance(scoring, list):
                for score in scoring:
                    sns.boxplot(
                        data=pd.DataFrame(
                            data=permutation_importances[estimator_name][score][
                                "importances"
                            ].T,
                            columns=self.feature_names,
                        ),
                        orient="h",
                        showmeans=True,
                        meanprops={"marker": "v", "markerfacecolor": "white"},
                    )
                    plt.title(
                        f"Permutation Importances ({estimator_name}, {str(score).upper()})",
                        fontsize=13,
                    )
                    plt.show()
            else:
                sns.boxplot(
                    data=pd.DataFrame(
                        data=permutation_importances[estimator_name]["importances"].T,
                        columns=self.feature_names,
                    ),
                    orient="h",
                    showmeans=True,
                    meanprops={"marker": "v", "markerfacecolor": "white"},
                )
                if scoring is not None:
                    plt.title(
                        f"Permutation Importances ({estimator_name}, {str(scoring).upper()})",
                        fontsize=13,
                    )
                else:
                    plt.title(
                        f"Permutation Importance ({estimator_name})",
                        fontsize=13,
                    )
                plt.show()

    def show_decision_tree(
        self,
        feature_names: list = None,
        class_names: Union[list, bool] = None,
        filepath: str = "decision_tree_viz",
    ):
        """
        DecisionTreeClassifier를 사용한 결과를 시각화한다.

        :param feature_names: list of str, default=None
            Names of each of the features.
            If None generic names will be used ("feature_0", "feature_1", ...).

        :param class_names: list of str or bool, default=None
            Names of each of the target classes in ascending numerical order.
            Only relevant for classification and not supported for multi-output.
            If ``True``, shows a symbolic representation of the class name.

        :param filepath: str, default="decision_tree_viz"
            The filepath to save the plot visualized by `DecisionTreeClassifier`.
        """

        if "DecisionTreeClassifier" in self.best_estimators.keys():
            os.makedirs(filepath, exist_ok=True)
            export_graphviz(
                decision_tree=self.best_estimators["DecisionTreeClassifier"],
                out_file=f"{filepath}/decision_tree.dot",
                class_names=class_names,
                feature_names=feature_names,
                impurity=True,
                filled=True,
            )
            with open(f"{filepath}/decision_tree.dot") as f:
                dot_graph = f.read()
            return graphviz.Source(dot_graph)
        else:
            raise ValueError("Only the decision tree model can be used.")

    def create_params_candidates(self, hyperparams_space: dict) -> dict:
        """
        알고리즘별 파라미터 조합 수 확인

        :param hyperparams_space: dictionary
        """

        candidates = {}
        for estimator_name, params_grid in self.generate_params_grid(
            hyperparams_space
        ).items():
            n = 1
            for param in params_grid.keys():
                try:
                    n *= len(params_grid[param])
                except:
                    continue
            candidates.update({estimator_name: n})
        return candidates
