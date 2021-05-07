from datetime import date
import pandas as pd
import numpy as np
from typing import Union
from scipy.sparse.construct import rand
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


def features_describe(
    x: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    dates: Union[pd.Series, np.ndarray] = None,
) -> pd.DataFrame:
    y_ = np.array(y)
    if dates is not None:
        dates = np.array(dates)
        result_dict = {
            "feature": [],
            "bin": [],
            "date": [],
            "cnt_bad": [],
            "cnt_good": [],
            "cnt_all": [],
            "pcnt_of_all": [],
            "bad_rate": [],
        }
    else:
        result_dict = {
            "feature": [],
            "bin": [],
            "cnt_bad": [],
            "cnt_good": [],
            "cnt_all": [],
            "pcnt_of_all": [],
            "bad_rate": [],
        }
    for feature in x.columns:
        for bin_value in x[feature].sort_values().unique():
            if dates is not None:
                for date in sorted(np.unique(dates)):
                    result_dict["feature"].append(feature)
                    result_dict["bin"].append(bin_value)
                    result_dict["date"].append(date)
                    cnt_all = len(
                        y_[
                            (np.isin(np.array(x[feature]), bin_value))
                            & (np.isin(dates, date))
                        ]
                    )
                    cnt_bad = y_[
                        (np.isin(np.array(x[feature]), bin_value))
                        & (np.isin(dates, date))
                    ].sum()
                    result_dict["cnt_bad"].append(cnt_bad)
                    result_dict["cnt_good"].append(cnt_all - cnt_bad)
                    result_dict["cnt_all"].append(cnt_all)
                    result_dict["pcnt_of_all"].append(cnt_all / len(y_))
                    try:
                        result_dict["bad_rate"].append(cnt_bad / cnt_all)
                    except ZeroDivisionError:
                        result_dict["bad_rate"].append(0)
            else:
                result_dict["feature"].append(feature)
                result_dict["bin"].append(bin_value)
                cnt_all = len(y_[np.isin(np.array(x[feature]), bin_value)])
                cnt_bad = y_[np.isin(np.array(x[feature]), bin_value)].sum()
                result_dict["cnt_bad"].append(cnt_bad)
                result_dict["cnt_good"].append(cnt_all - cnt_bad)
                result_dict["cnt_all"].append(cnt_all)
                result_dict["pcnt_of_all"].append(cnt_all / len(y_))
                try:
                    result_dict["bad_rate"].append(cnt_bad / cnt_all)
                except ZeroDivisionError:
                    result_dict["bad_rate"].append(0)

    result_df = pd.DataFrame.from_dict(result_dict)
    if dates is not None:
        result_df = pd.pivot_table(
            result_df,
            values=["cnt_all", "pcnt_of_all", "cnt_good", "cnt_bad", "bad_rate"],
            columns=["date"],
            index=["feature", "bin"],
        )
    return result_df


def data_stats(
    y: Union[pd.Series, np.ndarray],
    dates: Union[pd.Series, np.ndarray] = None,
) -> pd.DataFrame:
    y_ = np.array(y)
    if dates is not None:
        result_dict = {
            "date": [],
            "cnt_all": [],
            "cnt_good": [],
            "cnt_bad": [],
            "bad_rate": [],
        }
    else:
        result_dict = {"cnt_all": [], "cnt_good": [], "cnt_bad": [], "bad_rate": []}

    if dates is not None:
        dates = np.array(dates)
        for date in sorted(np.unique(dates)):
            result_dict["date"].append(date)
            cnt_all = len(y_[np.isin(dates, date)])
            result_dict["cnt_all"].append(cnt_all)
            cnt_bad = y_[np.isin(dates, date)].sum()
            result_dict["cnt_bad"].append(cnt_bad)
            result_dict["cnt_good"].append(cnt_all - cnt_bad)
            try:
                result_dict["bad_rate"].append(cnt_bad / cnt_all)
            except ZeroDivisionError:
                result_dict["bad_rate"].append(0)
    else:
        cnt_all = len(y_)
        result_dict["cnt_all"].append(cnt_all)
        cnt_bad = y_.sum()
        result_dict["cnt_bad"].append(cnt_bad)
        result_dict["cnt_good"].append(cnt_all - cnt_bad)
        try:
            result_dict["bad_rate"].append(cnt_bad / cnt_all)
        except ZeroDivisionError:
            result_dict["bad_rate"].append(0)
    result_df = pd.DataFrame.from_dict(result_dict)
    return result_df


def features_scoring(
    x: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    dates: Union[pd.Series, np.ndarray] = None,
    metric: str = "roc_auc",
    cv: int = 3,
    random_state: int = None,
    n_jobs: int = None,
    class_weight: str = None,
    use_gini: bool = False,
) -> pd.DataFrame:

    if dates is not None:
        dates = np.array(dates)
        result_dict = {
            "date": [],
            "feature": [],
            "score": [],
        }
    else:
        result_dict = {
            "feature": [],
            "score": [],
        }
    for feature in x.columns:
        if dates is not None:
            for date in sorted(np.unique(dates)):
                result_dict["date"].append(date)
                result_dict["feature"].append(feature)
                score = _calc_score(
                    x[np.isin(dates, date)][feature],
                    y[np.isin(dates, date)],
                    random_state=random_state,
                    class_weight=class_weight,
                    cv=cv,
                    metric=metric,
                    n_jobs=n_jobs,
                )
                if use_gini:
                    score = (2 * score - 1) * 100
                result_dict["score"].append(score)
        else:
            result_dict["feature"].append(feature)
            score = _calc_score(
                x[feature],
                y,
                random_state=random_state,
                class_weight=class_weight,
                cv=cv,
                metric=metric,
                n_jobs=n_jobs,
            )
            if use_gini:
                score = (2 * score - 1) * 100
            result_dict["score"].append(score)
    result_df = pd.DataFrame.from_dict(result_dict)
    if dates is not None:
        result_df = pd.pivot_table(
            result_df,
            values=["score"],
            columns=["date"],
            index=["feature"],
        )
    return result_df


def model_scoring(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    dates: Union[pd.Series, np.ndarray] = None,
    use_gini: bool = False,
) -> pd.DataFrame:
    if dates is not None:
        dates = np.array(dates)
        result_dict = {"date": [], "score": []}
        for date in sorted(np.unique(dates)):
            result_dict["date"].append(date)
            score = roc_auc_score(
                y_true[np.isin(dates, date)], y_pred[np.isin(dates, date)]
            )
            if use_gini:
                score = (2 * score - 1) * 100
            result_dict["score"].append(score)
    else:
        result_dict = {"score": []}
        score = roc_auc_score(y_true, y_pred)
        if use_gini:
            score = (2 * score - 1) * 100
        result_dict["score"].append(score)

    result_df = pd.DataFrame.from_dict(result_dict)
    return result_df


def calc_psi(x: pd.DataFrame, dates: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
    result_dict = {}
    for feature in x.columns:
        result_dict[feature] = []
        for date in np.delete(sorted(np.unique(dates)), 0):
            result_dict[feature].append(
                _calculate_psi_bins(
                    x[np.isin(dates, sorted(np.unique(dates))[0])][feature],
                    x[np.isin(dates, date)][feature],
                )
            )
    result_df = pd.DataFrame.from_dict(result_dict)
    result_df["date"] = np.delete(sorted(np.unique(dates)), 0)
    result_df = result_df.set_index("date")
    return result_df.T


def _calc_score(
    x: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    random_state: int = None,
    class_weight: str = None,
    cv: int = 3,
    metric: str = "roc_auc",
    n_jobs: int = None,
) -> float:
    model = LogisticRegression(
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=n_jobs,
    )
    scores = cross_val_score(
        model,
        x.values.reshape(-1, 1),
        y,
        cv=cv,
        scoring=metric,
        n_jobs=n_jobs,
    )
    return np.mean(scores)


def _calculate_psi_bins(expected: pd.DataFrame, actual: pd.DataFrame) -> float:
    initial_counts = expected.value_counts()
    new_counts = actual.value_counts()

    df = pd.DataFrame({"Initial_Count": initial_counts, "New_Count": new_counts})

    df["Initial_Percent"] = df["Initial_Count"] / np.sum(df["Initial_Count"])
    df["Initial_Percent"] = df["Initial_Percent"].apply(
        lambda x: 0.0001 if x == 0 else x
    )
    df["New_Percent"] = df["New_Count"] / np.sum(df["New_Count"])
    df["New_Percent"] = df["New_Percent"].apply(lambda x: 0.0001 if x == 0 else x)
    df["PSI"] = (df["New_Percent"] - df["Initial_Percent"]) * np.log(
        df["New_Percent"] / df["Initial_Percent"]
    )
    psi_values = np.sum(df["PSI"] * 100)
    return psi_values
