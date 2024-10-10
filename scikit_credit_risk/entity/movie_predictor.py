"""
author @ kumar dahal
this code is written to predict the output result
"""

import os
import sys

from movie_prediction.exception import MovieException

from movie_prediction.utils.common import load_object

import pandas as pd


class MovieData:

    def __init__(self,
                 originalTitle: str,
                 distributor: str,
                 opening_theaters: float,
                 budget: float,
                 MPAA: str,
                 release_days: float,
                 startYear: float,
                 runtimeMinutes: float,
                 genres_y: str,
                 averageRating: float,
                 numVotes: float,
                 ordering: float,
                 category: str,
                 primaryName: str

                 ):
        try:
            self.originalTitle = originalTitle
            self.distributor = distributor
            self.opening_theaters = opening_theaters
            self.budget = budget
            self.MPAA = MPAA
            self.release_days = release_days
            self.startYear = startYear
            self.runtimeMinutes = runtimeMinutes
            self.genres_y = genres_y
            self.averageRating = averageRating

            self.numVotes = numVotes
            self.ordering = ordering
            self.category = category
            self.primaryName = primaryName
        except Exception as e:
            raise MovieException(e, sys) from e

    def get_movies_input_data_frame(self):

        try:
            movies_input_dict = self.get_movies_data_as_dict()
            return pd.DataFrame(movies_input_dict)
        except Exception as e:
            raise MovieException(e, sys) from e

    def get_movies_data_as_dict(self):
        try:
            input_data = {
                "originalTitle": [self.originalTitle],
                "distributor": [self.distributor],
                "opening_theaters": [self.opening_theaters],
                "budget": [self.budget],
                "MPAA": [self.MPAA],
                "release_days": [self.release_days],
                "startYear": [self.startYear],
                "runtimeMinutes": [self.runtimeMinutes],
                "genres_y": [self.genres_y],
                "averageRating": [self.averageRating],
                "numVotes": [self.numVotes],
                "ordering": [self.ordering],
                "category": [self.category],
                "primaryName": [self.primaryName]}
            return input_data
        except Exception as e:
            raise MovieException(e, sys)


class MoviesPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise MovieException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise MovieException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            world_revenue = model.predict(X)
            return world_revenue
        except Exception as e:
            raise MovieException(e, sys) from e