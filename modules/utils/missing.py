"""
Create Missing Values in Data
"""

import enum
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


class SplitMode(enum.Enum):
    """
    Split mode for CreateMissingDataFrame
    """
    # Mode: Random, Linear
    Random = 'Random'
    Linear = 'Linear'

    @staticmethod
    def get_from_string(string: str) -> 'SplitMode':
        """
        Get SplitMode from string
        @param: string [str] - Charactor represent SplitMode
        @return: SplitMode instance [SplitMode]
        """
        if string == SplitMode.Random.value:
            return SplitMode.Random
        elif string == SplitMode.Linear.value:
            return SplitMode.Linear
        else:
            raise ValueError('Invalid split_mode')


class CreateMissingDataFrame:
    """
    To create missing dataframe, data is splited into `missing_gaps` parts.\n
    Each part will remove a similar amount of missing data. An mount defined by `missing_percentage`% of entire data.\n
    If `split_mode` is `SplitMode.Random`, each part will remove randomly on this part.\n
    If `split_mode` is `SplitMode.Linear`, each part will remove the center data on this part.\n
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        missing_percentage: float,
        missing_gaps: int,
        seed: int = None,
        split_mode: Union[SplitMode, str] = SplitMode.Linear,
        is_concate: bool = False,
        is_constant_missing: bool = False,
        safe_random_window: int = 0
    ):
        # Original Dataframe [pd.DataFrame]
        self.dataframe: pd.DataFrame = dataframe
        # Missing percentage for each gap [float]
        self.missing_percentage: float = missing_percentage
        # Amount of missing gaps [int]
        self.missing_gaps: int = missing_gaps
        # Seed for random [int]
        self.seed = seed
        # If split mode is string, convert to SplitMode instance [SplitMode]
        self.split_mode: SplitMode = SplitMode.get_from_string(
            split_mode) if isinstance(split_mode, str) else split_mode
        # Null value [pd.NA]
        self.empty_value = pd.NA
        # Dropped dataframe [pd.DataFrame]
        self.dropped_dataframe: pd.DataFrame = None
        # List of pair missing index [list[tuple[int, int]]]
        self.missing_indexs: list[tuple[int, int]] = None
        # If is_concate is True, remove missing values at other position and concate it [bool]
        self.is_concate: bool = is_concate
        # If is_constant_missing is True, missing_percentage will be constant [bool]
        self.is_constant_missing: bool = is_constant_missing
        # If safe_random_zone is True, random zone will be safe [bool]
        # Only use when split_mode is Random
        self.safe_random_window: int = safe_random_window

    def __dropping_dataframe(self):
        """
        When call method, implement dropping dataframe and assign to dropped_dataframe
        @param: None
        @return: None
        """
        # Set random seed of numpy. If seed is None, unset seed
        if self.seed is not None:
            np.random.seed(self.seed)
        else:
            np.random.default_rng(0)

        # Init list of missing index
        mising_indexs = []

        # Get amount of value can be missing
        # length of dataframe * missing_percentage / 100 [int]
        if self.is_constant_missing:
            missing_amount = int(self.missing_percentage)
        else:
            missing_amount = int(
                self.dataframe.shape[0] * self.missing_percentage / 100)

        # Copy dataframe to working_dataframe to prevent change original dataframe
        working_dataframe = self.dataframe.copy()

        # Length of each part for missing_gaps
        # length of dataframe / missing_gaps [int]
        part_index = int(self.dataframe.shape[0] / self.missing_gaps)

        # Split dataframe in Random mode
        # How it work:
        # - Define the dataframe that will be affected: splited_dataframe
        # - Get safe random zone: safe_random_zone
        #   Ex: if data has 1000 rows, missing_amount = 100, safe_random_zone = 900.
        #   In case random algorithm return 900, it will have enough space to drop 100 value
        # - Get random index: random_index
        # - Get missing index: upper_index, lower_index
        # - Add missing index to list
        # - Drop value by assign empty_value to dataframe
        if self.split_mode == SplitMode.Random:
            for n in range(self.missing_gaps):
                # Split dataframe
                splited_dataframe = working_dataframe[part_index *
                                                      n: part_index * (n + 1)]
                # Get random index
                safe_random_zone = splited_dataframe.shape[0] - missing_amount
                # Create random range
                random_upper_bound = 0 + self.safe_random_window
                random_lower_bound = safe_random_zone - self.safe_random_window
                # Check if safe_random_zone is valid
                if random_upper_bound > random_lower_bound:
                    raise ValueError(
                        f'Invalid safe_random_window. safe_random_window must be smaller than {safe_random_zone // 2}')
                # Get random index
                random_index = np.random.randint(
                    random_upper_bound, random_lower_bound)
                # Get missing index
                upper_index = part_index * n + random_index
                lower_index = upper_index + missing_amount
                # Add missing index to list
                mising_indexs.append((upper_index, lower_index))
                # Drop value
                working_dataframe[upper_index: lower_index] = self.empty_value

        # Split dataframe in Linear mode
        # How it work:
        # - Define offset: offset
        #   Offset is the begining index for missing data
        #   Ex: if data has 1000 rows, missing_amount = 100, offset = 450.
        #   Thus, missing data will be in range [450, 550] which is the middle of missing part
        # - Get missing index: upper_index, lower_index
        # - Add missing index to list
        # - Drop value by assign empty_value to dataframe
        elif self.split_mode == SplitMode.Linear:
            for n in range(self.missing_gaps):
                # Get missing index
                offset = part_index // 2 - missing_amount // 2
                upper_index = part_index * n + offset
                lower_index = upper_index + missing_amount
                # Add missing index to list
                mising_indexs.append((upper_index, lower_index))
                # Drop value
                working_dataframe[upper_index: lower_index] = self.empty_value

        # Save dropped dataframe and missing index
        self.dropped_dataframe = working_dataframe
        self.missing_indexs = mising_indexs

        logger.success(
            f'Dropped dataframe successfully. Missing indexs: {self.missing_indexs}')

    def __len__(self):
        return self.missing_gaps * 2

    def __getitem__(self, index):
        # If dropped dataframe is None, implement dropping dataframe
        if self.dropped_dataframe is None:
            self.__dropping_dataframe()

        indexes = self.missing_indexs[index // 2]

        # If index is even
        if index % 2 == 0:
            df = self.dataframe[:indexes[0]]
            mdf = self.dataframe[indexes[0]:indexes[1]]
            if self.is_concate:
                df = df.dropna()
            return df, mdf
        # If index is odd
        if index % 2 == 1:
            df = self.dataframe[indexes[1]:]
            mdf = self.dataframe[indexes[0]:indexes[1]]
            if self.is_concate:
                df = df.dropna()
            return df.iloc[::-1], mdf.iloc[::-1]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_dfs(self) -> list[tuple[pd.DataFrame]]:
        """
        Get list of missing dataframe and filled dataframe
        """
        dfs = []
        for _, df in enumerate(self):
            dfs.append(df)
        return dfs

    def plot(self, save_path: str = None):
        """
        Plot missing dataframe
        """
        if self.dropped_dataframe is None:
            self.__dropping_dataframe()

        plt.figure(figsize=(15, 5))
        plt.plot(self.dropped_dataframe)
        plt.title('Missing Dataframe')
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
