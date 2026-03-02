import pandas as pd
import numpy as np


class CustomData:
    """Collects form input and exposes a DataFrame for prediction.

    Fields match the usage in `app.py`.
    """

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        # Build a single-row DataFrame with predictable columns
        data = {
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [float(self.reading_score)],
            "writing_score": [float(self.writing_score)],
        }
        return pd.DataFrame(data)


class PredictPipeline:
    """A minimal prediction pipeline used for local dev/smoke tests.

    This implementation does not load a real model. Instead it produces a
    deterministic prediction based on the average of reading/writing scores.
    Replace with the real model-loading logic later.
    """

    def __init__(self):
        # placeholder for a real model (e.g., load from a pickle)
        self.model = None

    def predict(self, df: pd.DataFrame):
        # Ensure expected columns exist
        if not {"reading_score", "writing_score"}.issubset(df.columns):
            raise ValueError("DataFrame must contain reading_score and writing_score")

        avg = df[["reading_score", "writing_score"]].mean(axis=1)
        # Simple rule: average >= 50 => 'Pass' else 'Fail'
        preds = np.where(avg >= 50, "Pass", "Fail")
        return preds.tolist()
