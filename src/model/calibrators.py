import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def calibrate_platt(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(pred.reshape(-1, 1), y)
    return lr.predict_proba(pred.reshape(-1, 1))[:, 1]


def calibrate_isotonic(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pred, y)
    return iso.predict(pred)