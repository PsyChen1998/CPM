# 1 Features
This project refactors the CPM (*Connectome-based Predictive Modelling*) model:
1. The API is designed to be compatible with scikit-learn.
2. Added support for binary classification tasks (using logistic regression).
3. The code is very simple, allowing users to create new models according to their needs.

# 2 CPM
CPM consists of three parts:
1. **Feature Selection**: Masks are created based on the p-value. Features with a value greater than the p-value are included in the positive mask, while those with a value less than -p are included in the negative mask. The p-value can be treated as a hyperparameter.
2. **Feature Calculation**: All selected features are summed to create a single variable.
3. **Model Fitting**: Models are fitted for positive, negative, and both masks. The "both" model is composed of the positive and negative variables, resulting in two variables.
4. **Prediction**: Predictions are made using the fitted models.

# 3 Contributions
- **[Mingzhe Zhang](https://github.com/psyMingzheZhang)** authored [`CPM/1_original_version.ipynb`](https://github.com/PsyChen1998/CPM/blob/main/CPM/1_original_version.ipynb), which use the codes from the [original version](https://github.com/YaleMRRC/CPM).
- **[Guoqiu Chen](https://github.com/PsyChen1998)** developed the CPM package and validated it using HBN data in [`CPM/2_new_version.ipynb`](https://github.com/PsyChen1998/CPM/blob/main/CPM/2_new_version.ipynb), which also serves as example code.

# 4 Reference
Shen, X., Finn, E. S., Scheinost, D., Rosenberg, M. D., Chun, M. M., Papademetris, X., & Constable, R. T. (2017). Using connectome-based predictive modeling to predict individual behavior from brain connectivity. _nature protocols_, _12_(3), 506-518.
