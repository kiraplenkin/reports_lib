import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target, unique_labels
from typing import Union, Tuple
from .functions import (
    features_describe,
    data_stats,
    features_scoring,
    model_scoring,
    calc_psi,
)


class ReportBuilder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        check_corr: bool = False,
        check_features_stats: bool = False,
        check_stats: bool = False,
        check_features_scoring: bool = False,
        check_model_score: bool = None,
        check_psi: bool = False,
        abs_corr: bool = False,
        method_corr: str = "pearson",
        metric: str = "roc_auc",
        cv: int = 3,
        random_state: int = None,
        n_jobs: int = None,
        class_weight: str = None,
        use_gini: bool = False,
    ) -> None:
        self.check_corr = check_corr
        self.check_features_stats = check_features_stats
        self.check_stats = check_stats
        self.check_features_scoring = check_features_scoring
        self.check_model_score = check_model_score
        self.check_psi = check_psi
        self.abs_corr = abs_corr
        self.method_corr = method_corr
        self.metric = metric
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.use_gini = use_gini

        self.corr = None
        self.psi = None
        self.features_stats = None
        self.features_stats_on_time = None
        self.stats = None
        self.features_score = None
        self.features_score_on_time = None
        self.model_score = None
        self.model_score_on_time = None
        self.psi = None

        self.reports = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray] = None,
        dates: Union[pd.Series, np.ndarray] = None,
    ) -> None:
        X, y = self._check_inputs(X, y)
        self.classes_ = unique_labels(y)
        self.dates = dates
        self.X_ = X
        self.y_ = y
        self.y_pred = y_pred

        if self.check_corr:
            self.corr = self.X_.corr(method=self.method_corr)
            if self.abs_corr:
                self.corr = np.abs(self.corr)
            self.corr.name = "corr"
            self.reports.append(self.corr)

        if self.check_features_stats:
            self.features_stats = features_describe(self.X_, self.y_, dates=None)
            self.features_stats.name = "features_stats"
            self.reports.append(self.features_stats)

            if self.dates is not None:
                self.features_stats_on_time = features_describe(
                    self.X_, self.y_, self.dates
                )
                self.features_stats_on_time.name = "features_stats_on_time"
                self.reports.append(self.features_stats_on_time)

        if self.check_stats:
            if self.dates is not None:
                self.stats = data_stats(self.y_, self.dates)
            else:
                self.stats = data_stats(self.y_)
            self.stats.name = "stats"
            self.reports.append(self.stats)

        if self.check_features_scoring:
            self.features_score = features_scoring(
                self.X_,
                self.y_,
                dates=None,
                metric=self.metric,
                cv=self.cv,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                class_weight=self.class_weight,
                use_gini=self.use_gini,
            )
            self.features_score.name = "features_score"
            self.reports.append(self.features_score)

            if self.dates is not None:
                self.features_score_on_time = features_scoring(
                    self.X_,
                    self.y_,
                    dates=self.dates,
                    metric=self.metric,
                    cv=self.cv,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    class_weight=self.class_weight,
                    use_gini=self.use_gini,
                )
                self.features_score_on_time.name = "features_score_on_time"
                self.reports.append(self.features_score_on_time)

        if self.check_model_score:
            if self.y_pred is None:
                raise "Required y_pred vector!"
            self.model_score = model_scoring(
                self.y_, self.y_pred, dates=None, use_gini=self.use_gini
            )
            self.model_score.name = "model_score"
            self.reports.append(self.model_score)

            if self.dates is not None:
                self.model_score_on_time = model_scoring(
                    self.y_, self.y_pred, dates=self.dates, use_gini=self.use_gini
                )
                self.model_score_on_time.name = "model_score_on_time"
                self.reports.append(self.model_score_on_time)

        if self.check_psi:
            if self.dates is None:
                raise "Required dates vector!"
            self.psi = calc_psi(self.X_, self.dates)
            self.psi.name = "psi"
            self.reports.append(self.psi)

        return self

    def to_excel(self, file_name: str = "") -> None:
        writer = pd.ExcelWriter(file_name, engine="xlsxwriter")
        for report in self.reports:
            if report is not None:
                report.to_excel(writer, sheet_name=report.name)
        writer.save()

    def _check_inputs(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]:
        """
        Check input data
        :param X: data matrix
        :param y: target vector
        :return: X, y
        """
        if type_of_target(y) != "binary":
            raise ValueError("y vector should be binary")

        return X, y
