from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
from scipy.stats import spearmanr, pointbiserialr
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import pandas as pd
from typing import Literal


class CPMBase(BaseEstimator):
    """
    A base class for Connectome-based Predictive Modeling (CPM) in either regression or classification tasks.

    Parameters
    ----------
    method : Literal['pearson', 'spearman', 'pointbiserial'], default='pearson'
        The correlation method to use for feature selection:
        - 'pearson': Pearson correlation coefficient.
        - 'spearman': Spearman rank-order correlation coefficient.
        - 'pointbiserial': Point-biserial correlation coefficient.

    threshold : float, default=0.05
        The correlation threshold to determine which features are selected. Features with an absolute correlation
        value greater than this threshold are considered for the model.

    Attributes
    ----------
    method : str
        The correlation method used for feature selection.

    threshold : float
        The correlation threshold used for feature selection.

    mask_positive : np.ndarray or None
        Boolean mask array indicating the features with positive correlations above the threshold.

    mask_negative : np.ndarray or None
        Boolean mask array indicating the features with negative correlations below the threshold.

    mask_both : np.ndarray or None
        Boolean mask array indicating the features with absolute correlations below the threshold.

    Methods
    -------
    _make_mask(X, y):
        Computes the correlation between each feature in X and the target y, then creates masks based on the specified
        correlation method and threshold.

    fit(X, y):
        Fits the CPM model to the provided data X and target y using the selected correlation method and threshold.

    predict(X):
        Predicts the target variable using the fitted models based on the provided data X.

"""

    def __init__(self,
                 method: Literal['pearson', 'spearman', 'pointbiserial'] = 'pearson',
                 threshold: float = 0.05,
                 to_predict: Literal['positive', 'negative', 'both'] or None = None, ):

        self.method = method
        self.threshold = threshold
        self.to_predict = to_predict
        self.mask = dict()
        self._model = dict(positive=self._model_function(),
                          negative=self._model_function(),
                          both=self._model_function())

    def _make_mask(self, X, y):
        """
        Computes correlation masks for the features based on the correlation with the target y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,)
            The target variable.

        Returns
        -------
        None
        """
        X, y = check_X_y(X, y)
        # 1. get corr
        if self.method == 'pearson':
            corr = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
        elif self.method == 'spearman':
            corr = [spearmanr(X[:, i], y)[0] for i in range(X.shape[1])]
        elif self.method == 'pointbiserial':
            corr = [pointbiserialr(X[:, i], y)[0] for i in range(X.shape[1])]
        corr = np.array(corr)

        # 2. make masks
        self.mask['positive'] = corr > self.threshold
        self.mask['negative'] = corr < -self.threshold

    def make_masked_data(self, X):
        """
        Applies masking to the input data and returns the masked data.

        This method processes the input data matrix `X` by applying pre-defined masks
        for 'positive' and 'negative' features. It calculates the sum of the features
        specified in the 'positive' and 'negative' masks for each instance in `X`,
        and returns these sums in a dictionary with the following keys:

        - `positive`: A numpy array with the sum of 'positive' masked features for each instance,
          reshaped to a column vector of shape (n, 1), where `n` is the number of instances.
        - `negative`: A numpy array with the sum of 'negative' masked features for each instance,
          reshaped to a column vector of shape (n, 1).
        - `both`: A 2D numpy array with two columns, where the first column contains the sums
          of 'positive' masked features and the second column contains the sums of 'negative'
          masked features for each instance.

        Parameters:
        -----------
        X : numpy.ndarray
            The input data matrix of shape (n_samples, n_features), where each row represents
            an instance and each column represents a feature.

        Returns:
        --------
        X_masked : dict
            A dictionary containing the masked data arrays for 'positive', 'negative', and 'both'.
            Each key corresponds to a numpy array as described above.
        """
        n = len(X)
        X_positive = X[:, self.mask['positive']].sum(axis=1)
        X_negative = X[:, self.mask['negative']].sum(axis=1)
        X_both = np.array([X_positive, X_negative]).T
        X_maksed = dict(positive=X_positive.reshape(n, 1), negative=X_negative.reshape(n, 1), both=X_both)
        return X_maksed

    def fit(self, X, y):
        """
        Fits the CPM model by selecting features based on correlation with the target variable and fitting models to
        the selected features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,)
            The target variable.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.array(X)
        self._make_mask(X, y)
        X_maksed = self.make_masked_data(X)

        for key, this_X in X_maksed.items():
            model = self._model[key]
            model.fit(this_X, y)

    def predict(self, X):
        """
        Predicts the target variable using the models trained on positively and negatively correlated features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        pd.DataFrame
            A DataFrame with predictions from the models trained on positively correlated features, negatively correlated
            features, and both combined.
        """
        X = np.array(X)
        X_maksed = self.make_masked_data(X)

        result = dict()
        for name, model in self._model.items():
            this_X = X_maksed[name]
            result[name] = model.predict(this_X)

        if self.to_predict is not None:
            return result[self.to_predict]
        else:
            return result

class CPMRegressor(CPMBase):
    _model_function = LinearRegression

class CPMClassifier(CPMBase):
    _model_function = LogisticRegression
