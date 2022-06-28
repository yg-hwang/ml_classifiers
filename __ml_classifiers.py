import os
import graphviz
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings

warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

from joblib import dump
from typing import Union
from tempfile import mkdtemp
from datetime import datetime

cache_dir = mkdtemp()

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
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


class Classifiers(object):
    def __init__(self, feature_scaler: str = None, random_state: int = 123):
        """
        분류 모델별 예측 시 사용할 Feature scaler를 지정한다.

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

    def transform_feature_scale(self):
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

    def separate_by_feature_and_label(self, data: pd.DataFrame, target: str) -> tuple:
        """
        모델링에 사용할 데이터를 Feature(X)와 Target(y)으로 분리한다.

        :param data: pd.DataFrame
            Dataset to fit and train model.

        :param target: str
            A target column of dataset to try to predict.
        """

        features = data.loc[:, data.columns != target].values
        label = data[target].values
        self.label_count = data[target].nunique()

        if self.transform_feature_scale() is None:
            return features, label
        else:
            scaler = self.transform_feature_scale()
            scaler.fit(features)
            features_scaled = scaler.transform(features)
            return features_scaled, label

    def split_data_into_train_test(
        self,
        data: pd.DataFrame,
        target: str,
        test_size: float = None,
        shuffle: bool = True,
    ) -> tuple:
        """
        모델링에 사용할 훈련, 검증 데이터를 분리한다.

        :param data: pd.DataFrame
            Dataset to fit and train model.

        :param target: str
            A target column of dataset to try to predict.

        :param test_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size.

        :param shuffle: bool, default=True
            Whether or not to shuffle the data before splitting. If shuffle=False
            then stratify must be None.
        """

        features, label = self.separate_by_feature_and_label(data=data, target=target)
        return train_test_split(
            features,
            label,
            test_size=test_size,
            shuffle=shuffle,
            random_state=self.random_state,
            stratify=label,
        )

    def get_kfold_function(self, kfold: str, n_splits: int = 5, n_repeats: int = 10):
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

    def get_classifier_models(self) -> dict:
        """
        분류 모델에 사용할 알고리즘을 사전에 정의한다.
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

    def _fit_params(self, estimator_name: str) -> dict:
        """
        XGBoost, LightGBM으로 모델링 할 때 `early_stopping_rounds`, `eval_metric` 사용을 위한 함수
        """

        if estimator_name == "XGBClassifier":
            params = {"early_stopping_rounds": 100, "verbose": False}
            if self.label_count == 2:
                params["eval_metric"] = "logloss"
                return params
            else:
                params["eval_metric"] = "mlogloss"
                return params

        if estimator_name == "LGBMClassifier":
            params = {
                "callbacks": [early_stopping(100), log_evaluation(100)],
            }
            if self.label_count == 2:
                params["eval_metric"] = "binary_logloss"
                return params
            else:
                params["eval_metric"] = "multi_logloss"
                return params

    def run_cross_validation(
        self,
        data: pd.DataFrame,
        target: str,
        estimators: Union[str, list] = "all",
        kfold: str = None,
        n_splits: int = None,
        n_repeats: int = None,
        scoring: str = None,
        fit_params: dict = None,
    ) -> pd.DataFrame:
        """
        모델별 예측을 수행하여 그 결과를 반환한다.

        :param data: pd.DataFrame
            Dataset to fit and train model.

        :param target: str
            A target column of dataset to try to predict.

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

        :param fit_params: dict
            Parameters to pass to the fit method of the estimator.
        """

        if estimators == "all":
            classifiers = self.get_classifier_models()
        else:
            estimators_intersection = list(
                set(estimators) & set(self.get_classifier_models().keys())
            )
            classifiers = {}
            for label in estimators_intersection:
                classifiers.update({label: self.get_classifier_models()[label]})

        if fit_params is not None:
            if len(list(set(fit_params.keys()) & set(classifiers.keys()))) == 0:
                raise ValueError(
                    " If you use `fit_params`, "
                    "you must also include estimator in the list of `estimators`. "
                )

        print(" >>> Check performance metric by estimator. ")
        features, label = self.separate_by_feature_and_label(data=data, target=target)
        self.df_cv_result = pd.DataFrame()

        for estimator_name, estimator in classifiers.items():
            print(f" >>> {estimator_name} running... ")
            fit_start_time = datetime.now()

            if fit_params is None:
                params = None
            else:
                if estimator_name in fit_params.keys():
                    params = fit_params[estimator_name]
                else:
                    params = None

            if self.label_count > 2:
                cv_dict = cross_validate(
                    estimator=estimator() if params is None else estimator(**params),
                    X=features,
                    y=label,
                    cv=self.get_kfold_function(
                        kfold=kfold, n_splits=n_splits, n_repeats=n_repeats
                    ),
                    scoring=scoring,
                    n_jobs=-1,
                )
                df_cv = pd.DataFrame(data=cv_dict)
                df_cv["scoring"] = scoring
                df_cv["estimator_name"] = estimator_name
                self.df_cv_result = pd.concat(
                    [self.df_cv_result, df_cv], ignore_index=True
                )

            elif self.label_count == 2:
                for scoring in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                    cv_dict = cross_validate(
                        estimator=estimator()
                        if params is None
                        else estimator(**params),
                        X=features,
                        y=label,
                        cv=self.get_kfold_function(
                            kfold=kfold, n_splits=n_splits, n_repeats=n_repeats
                        ),
                        scoring=scoring,
                        n_jobs=-1,
                    )
                    df_cv = pd.DataFrame(data=cv_dict)
                    df_cv["scoring"] = scoring
                    df_cv["estimator_name"] = estimator_name
                    self.df_cv_result = pd.concat(
                        [self.df_cv_result, df_cv], ignore_index=True
                    )

            else:
                raise ValueError(" Single class can not create confusion matrix. ")
            print(
                f" Finished. (elapsed_time: {(datetime.now() - fit_start_time).seconds}s) "
            )
            print(
                " ------------------------------------------------------------------------------ "
            )
        return self.df_cv_result

    def show_cross_validation_result(self):
        """
        run_cross_validation()를 수행한 결과를 시각화할 때 사용한다.
        """

        df_cv_result_agg = (
            self.df_cv_result.groupby(["estimator_name", "scoring"])
            .agg(
                mean_test_score=("test_score", "mean"),
                mean_fit_time=("fit_time", "mean"),
            )
            .reset_index()
        )

        df_cv_result_ = self.df_cv_result.merge(
            right=df_cv_result_agg, on=["estimator_name", "scoring"], how="inner"
        ).reset_index(drop=True)

        df_cv_result_ = df_cv_result_.sort_values(
            by=["scoring", "mean_test_score", "estimator_name"],
            ascending=[True, False, True],
            ignore_index=True,
        )

        sns.set(rc={"figure.figsize": (12, 6)})
        for scoring in df_cv_result_["scoring"].unique():
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.4)
            sns.boxplot(
                data=df_cv_result_[df_cv_result_["scoring"] == scoring],
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
                data=df_cv_result_[df_cv_result_["scoring"] == scoring],
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
            ax1.set_title(scoring.upper(), fontsize=14)
            ax2.set_title("fit time (sec)", fontsize=14)

            df_tmp = df_cv_result_.loc[
                df_cv_result_["scoring"] == scoring,
                ["estimator_name", "mean_test_score", "mean_fit_time"],
            ].drop_duplicates(ignore_index=True)

            for index, row in df_tmp.iterrows():
                ax1.text(
                    x=row.mean_test_score,
                    y=index,
                    s=round(row.mean_test_score, 4),
                    color="black",
                    ha="center",
                    fontsize=13,
                )
                ax2.text(
                    x=row.mean_fit_time,
                    y=index,
                    s=round(row.mean_fit_time, 4),
                    color="black",
                    ha="center",
                    fontsize=13,
                )
            plt.show()

    def generate_params_grid(self, hyperparams_space: dict) -> dict:
        """
        하이퍼파라미터 튜닝을 시도할 모델과 해당 파라미터 값을 반환한다.

        :param hyperparams_space: dictionary
            사용자가 직접 정의한 모델과 파라미터 값의 dictionary
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

    def create_params_candidates(self, hyperparams_space: dict) -> dict:
        """
        모델 별 파라미터 조합 수 확인

        :param hyperparams_space: dictionary
            사용자가 직접 정의한 모델과 파라미터 값의 dictionary
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

    def search_hyperparameter(
        self,
        data: pd.DataFrame,
        target: str,
        hyperparams_space: dict,
        search_method: str = "random",
        test_size: float = 0.25,
        shuffle: bool = True,
        kfold: str = None,
        n_splits: int = None,
        n_repeats: int = None,
        scoring: str = None,
        n_iter: int = 50,
        factor: int = 3,
    ) -> pd.DataFrame:
        """
        :param data: pd.DataFrame
            Dataset to fit and train model.

        :param target: str
            A target column of dataset to try to predict.

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

        :param n_iter, default=5
            Number of parameter settings that are produced.

        :param factor: int or float, default=3
            The 'halving' parameter, which determines the proportion of candidates
            that are selected for each subsequent iteration. For example,
            ``factor=3`` means that only one third of the candidates are selected.
        """

        self.feature_names = data.loc[:, data.columns != target].columns
        self.df_hyperparams_search_result = pd.DataFrame()
        self.best_estimators = dict()
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.scoring = scoring

        print(" >>> Start to search best parameters by cross validation ")
        X_train, X_test, y_train, y_test = self.split_data_into_train_test(
            data=data, target=target, test_size=test_size, shuffle=shuffle
        )

        classifier_all = self.get_classifier_models().keys()
        params_grid = self.generate_params_grid(hyperparams_space=hyperparams_space)
        classifier_to_use = params_grid.keys()

        comm_classifiers = classifier_all & classifier_to_use
        classifiers = {}
        for label in comm_classifiers:
            classifiers.update({label: self.get_classifier_models()[label]})

        for estimator_name, estimator in classifiers.items():
            print(f" >>> {estimator_name} running... ")
            fit_start_time = datetime.now()

            if search_method == "grid":
                if (estimator_name == "XGBClassifier") | (
                    estimator_name == "LGBMClassifier"
                ):
                    model = GridSearchCV(
                        estimator=estimator(),
                        param_grid=params_grid[estimator_name],
                        fit_params=self._fit_params(estimator_name=estimator_name),
                        cv=self.get_kfold_function(
                            kfold=kfold, n_splits=n_splits, n_repeats=n_repeats
                        ),
                        scoring=scoring,
                        n_jobs=-1,
                        return_train_score=True,
                    )
                    model.fit(X_train, y_train)

                else:
                    model = GridSearchCV(
                        estimator=estimator(),
                        param_grid=params_grid[estimator_name],
                        cv=self.get_kfold_function(
                            kfold=kfold, n_splits=n_splits, n_repeats=n_repeats
                        ),
                        scoring=scoring,
                        n_jobs=-1,
                        return_train_score=True,
                    )
                    model.fit(X_train, y_train)

            elif search_method == "random":
                model = RandomizedSearchCV(
                    estimator=estimator(),
                    param_distributions=params_grid[estimator_name],
                    cv=self.get_kfold_function(
                        kfold=kfold, n_splits=n_splits, n_repeats=n_repeats
                    ),
                    scoring=scoring,
                    n_iter=n_iter,
                    n_jobs=-1,
                    return_train_score=True,
                )
                if (estimator_name == "XGBClassifier") or (
                    estimator_name == "LGBMClassifier"
                ):
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_test, y_test)],
                        **self._fit_params(estimator_name=estimator_name),
                    )
                else:
                    model.fit(X_train, y_train)

            elif search_method == "grid_halving":
                if (estimator_name == "XGBClassifier") or (
                    estimator_name == "LGBMClassifier"
                ):
                    model = HalvingGridSearchCV(
                        estimator=estimator(),
                        param_grid=params_grid[estimator_name],
                        cv=self.get_kfold_function(
                            kfold=kfold, n_splits=n_splits, n_repeats=n_repeats
                        ),
                        scoring=scoring,
                        n_jobs=-1,
                        return_train_score=True,
                        min_resources="exhaust",
                        factor=factor,
                    )
                    model.fit(X_train, y_train)
                else:
                    model = HalvingGridSearchCV(
                        estimator=estimator(),
                        param_grid=params_grid[estimator_name],
                        cv=self.get_kfold_function(
                            kfold=kfold, n_splits=n_splits, n_repeats=n_repeats
                        ),
                        scoring=scoring,
                        n_jobs=-1,
                        return_train_score=True,
                        min_resources="exhaust",
                        factor=factor,
                    )
                    model.fit(X_train, y_train)

            elif search_method == "random_halving":
                model = HalvingRandomSearchCV(
                    estimator=estimator(),
                    param_distributions=params_grid[estimator_name],
                    cv=self.get_kfold_function(
                        kfold=kfold, n_splits=n_splits, n_repeats=n_repeats
                    ),
                    scoring=scoring,
                    n_jobs=-1,
                    return_train_score=True,
                    n_candidates="exhaust",
                    factor=factor,
                )
                if (estimator_name == "XGBClassifier") or (
                    estimator_name == "LGBMClassifier"
                ):
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_test, y_test)],
                        **self._fit_params(estimator_name=estimator_name),
                    )
                else:
                    model.fit(X_train, y_train)

            else:
                raise ValueError(
                    "`search_method` should be ond of 'grid', 'random', 'grid_halving' or 'random_halving'."
                )

            print(
                f" Finished. (elapsed_time: {(datetime.now() - fit_start_time).seconds}s) "
            )
            print(f" Best Parameters: {model.best_params_} ")
            print(
                "------------------------------------------------------------------------------"
            )
            self.best_estimators.update({estimator_name: model.best_estimator_})
            os.makedirs("model_saved", exist_ok=True)
            dump(model.best_estimator_, f"model_saved/{estimator_name}.joblib")

            cv_result_tmp = pd.DataFrame(data=model.cv_results_)
            cv_result_tmp["estimator_name"] = estimator_name
            self.df_hyperparams_search_result = pd.concat(
                [self.df_hyperparams_search_result, cv_result_tmp], ignore_index=True
            )
        return self.df_hyperparams_search_result

    def show_hyperparameter_search_result(self):
        """
        search_hyperparameter()를 수행한 결과를 시각화할 때 사용한다.
        """

        df_hyperparams_search_result_ = (
            self.df_hyperparams_search_result[
                self.df_hyperparams_search_result.rank_test_score == 1
            ]
            .groupby("estimator_name")
            .head(1)
        )

        df_hyperparams_search_result_ = df_hyperparams_search_result_.sort_values(
            by="mean_test_score", ascending=False, ignore_index=True
        )

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
                    df_hyperparams_search_result_[
                        [f"split{i}_test_score", "estimator_name"]
                    ].rename(columns={f"split{i}_test_score": "test_score"}),
                ],
                ignore_index=True,
            )

        sns.set(rc={"figure.figsize": (12, 6)})
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
            data=df_hyperparams_search_result_,
            x="mean_fit_time",
            y="estimator_name",
            ax=ax2,
        )
        for ax in [ax1, ax2]:
            ax.tick_params(axis="x", labelsize=14)
            ax.tick_params(axis="y", labelsize=14)
            ax.set_xlabel(xlabel=None)
            ax.set_ylabel(ylabel=None)
        ax1.set_title(self.scoring.upper(), fontsize=14)
        ax2.set_title("mean fit time (sec)", fontsize=14)

        for index, row in df_hyperparams_search_result_.iterrows():
            ax1.text(
                x=row.mean_test_score,
                y=index,
                s=round(row.mean_test_score, 4),
                color="black",
                ha="center",
                fontsize=13,
            )
            ax2.text(
                x=row.mean_fit_time,
                y=index,
                s=round(row.mean_fit_time, 4),
                color="black",
                ha="center",
                fontsize=13,
            )

    def evaluate_model(self, y_test, pred, pred_proba=None) -> Union[tuple, float]:
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
            return accuracy, precision, recall, f1, roc_auc

    def show_precision_recall_by_thresholds(
        self, estimators: str, data: pd.DataFrame, target: str
    ):
        """
        threshold에 따른 이진 분류 성능 확인을 위한 정밀도, 재현율 확인
        """

        from sklearn.metrics import precision_recall_curve

        X_train, X_test, y_train, y_test = self.split_data_into_train_test(
            data=data, target=target
        )

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
        아래의 기준에 따라 best 모델로 선정할 파라미터 반환
        1) mean_test_score 높은순
        2) std_test_score 낮은순
        3) mean_fit_time 낮은순
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

    def get_best_classifer(self, estimator_name: str = None):
        """
        데이터 예측에 사용할 최종 estimator 반환.
        """
        if estimator_name is None:
            print(" >>> Predict from new data. ")
            return self.best_estimators[self.get_best_model_info()["estimator_name"]]
        else:
            print(" >>> Predict from new data. ")
            return self.best_estimators[estimator_name]

    def show_feature_importances(
        self, estimators: list = None, index: list = None, n_features: int = 10
    ):
        """
        훈련 데이터로 생성된 중요도이므로 이것에 너무 의존하지 않을 필요가 있음.
        """
        feature_importances = {}
        n_features = n_features
        sns.set(rc={"figure.figsize": (12, 6)})

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
        self,
        data: pd.DataFrame,
        target: str,
        estimators: list = None,
        scoring: str = None,
        index: list = None,
    ):
        """
        1) 기존 test data에서 하나의 feature에 대해 row 순서를 무작위로 섞어서 새로운 test data를 생성함.
        2) 새로운 test data로 score를 측정함.
        3) 기존 test data에 의한 score 대비 새로운 test data에 의한 score를 비교함.
        4) 만약 score가 많이 감소한다면 해당 feature가 중요하다는 의미임.
        5) 혹은 score가 거의 그대로라면 해당 feature는 그닥 중요하진 않다는 의미임.
        6) 이를 통해 score가 감소하지 않는 feature를 발견한다면, 모델 튜닝 시 해당 feature를 제와하여 수행함. (성능이 올라갈 수도 있는 가능성과 리소스 확보 차원에서 도움이 될 수 있음.)
        """
        X_train, X_test, y_train, y_test = self.split_data_into_train_test(
            data=data, target=target
        )

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

        sns.set(rc={"figure.figsize": (12, 12)})
        for estimator_name in permutation_importances.keys():
            if index is None:
                sns.boxplot(
                    data=pd.DataFrame(
                        data=permutation_importances[estimator_name]["importances"].T,
                        columns=self.feature_names,
                    ),
                    orient="h",
                    showmeans=True,
                    meanprops={"marker": "v", "markerfacecolor": "white"},
                )
                plt.title(f"permutation importances ({estimator_name})", fontsize=13)
                plt.show()
            else:
                sns.boxplot(
                    data=pd.DataFrame(
                        data=permutation_importances[estimator_name]["importances"].T,
                        columns=index,
                    ),
                    orient="h",
                    showmeans=True,
                    meanprops={"marker": "v", "markerfacecolor": "white"},
                )
                plt.title(f"permutation importances ({estimator_name})", fontsize=13)
                plt.show()

    def show_decision_tree(self, feature_names: list = None, class_names=None):
        """
        Decision Tree를 시각화 할 때 사용.
        """
        if "DecisionTreeClassifier" in self.best_estimators.keys():
            os.makedirs("viz", exist_ok=True)
            export_graphviz(
                decision_tree=self.best_estimators["DecisionTreeClassifier"],
                out_file="viz/tree.dot",
                class_names=class_names,
                feature_names=feature_names,
                impurity=True,
                filled=True,
            )
            with open("viz/tree.dot") as f:
                dot_graph = f.read()
            return graphviz.Source(dot_graph)
        else:
            raise ValueError("Only the decision tree model can be used.")
