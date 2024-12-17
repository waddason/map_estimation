
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def get_estimator():
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )

    return pipe
