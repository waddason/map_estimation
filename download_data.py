from pathlib import Path

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

DATA_PATH = Path('data')


if __name__ == '__main__':
    if not DATA_PATH.exists():
        DATA_PATH.mkdir()

    # Load the data
    print('Loading the data...', end='', flush=True)
    X_df, y = load_digits(return_X_y=True, as_frame=True)
    X_df['target'] = y

    X_train, X_test = train_test_split(
        X_df, test_size=0.2, random_state=57, shuffle=True,
        stratify=y
    )

    # Save the data
    X_train.to_csv(DATA_PATH / 'X_train.csv', index=False)
    X_test.to_csv(DATA_PATH / 'X_test.csv', index=False)
    print('done')
