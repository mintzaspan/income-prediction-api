# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import set_config
import pickle
import os

set_config(transform_output="pandas")


def split_data(df, target, test_size=0.2):
    """Split dataset to X_train, X_test, y_train, y_test.

    Args:
        df : Pandas DataFrame
        target : name of target variable
        test_size : percentage of data to use for testing

    Returns:
        X_train : predictors df for train
        X_test : predictors df for test
        y_train : target variable df for train
        y_test : target variable df for testing
    """

    # split to predictors and target
    y = df[target]
    X = df.drop(columns=[target])

    # split to train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0, stratify=y)

    return (X_train, X_test, y_train, y_test)


def train_model(X, y, categorical_features, numeric_features):
    """Trains a binary classification model on data provided and returns model object.

    Args:
        X : Pandas DataFrame including predictors
        y : Pandas Series or DataFrame including target variable only
        categorical_features: list of categorical columns in df
        numerical_features: list of numerical columns in df

    Returns:
        model : trained classifier
    """

    # Pre-processing
    # numeric transformer
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # categorical transformee
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", TargetEncoder(target_type='binary', random_state=0)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Classifier
    classifier = GradientBoostingClassifier(
        min_samples_leaf=20,
        max_depth=6,
        random_state=0,
        validation_fraction=0.1
    )

    # Full pipeline
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    pipe.fit(X, y)

    return (pipe)


def save_model(model):
    """Saves trained model in model directory.

    Args:
        model : trained sklearn model or pipeline

    Returns:
        None
    """

    # check if model_dir exists
    model_dir = 'model/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # save model
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_path):
    """Loads a trained model from a pickle object

    Args:
        model_path : path to .pkl file

    Returns:
        model : trained sklearn model or pipeline
    """

    with open(model_path, 'rb') as fp:
        model = pickle.load(fp)

    return (model)


if __name__ == "__main__":

    # load the data
    data = pd.read_csv(
        filepath_or_buffer='data/census.csv'
    )

    # split to predictors and target
    X_train, X_test, y_train, y_test = split_data(
        df=data,
        target='salary'
    )

    # declare column types
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    num_features = [
        "age",
        "fnlgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week"
    ]

    # train model
    model = train_model(
        X=X_train,
        y=y_train,
        categorical_features=cat_features,
        numeric_features=num_features
    )

    # save model
    save_model(model)

    # load model
    m = load_model('model/model.pkl')
