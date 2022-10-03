"""
Univariate Sine Wave
-----------------
"""

import numpy as np
from oats.generator._base import Generator


def _series_segmentation(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def _sine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    # timestamp = np.linspace(0, 10, length)
    timestamp = np.arange(length)
    value = np.sin(2 * np.pi * freq * timestamp)
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp * noise
    value = coef * value + offset
    return value


def _square_sine(level=5, length=500, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    value = np.zeros(length)
    for i in range(level):
        value += (
            1
            / (2 * i + 1)
            * _sine(
                length=length,
                freq=freq * (2 * i + 1),
                coef=coef,
                offset=offset,
                noise_amp=noise_amp,
            )
        )
    return value


def _collective_global_synthetic(length, base, coef=1.5, noise_amp=0.005):
    value = []
    norm = np.linalg.norm(base)
    base = base / norm
    num = int(length / len(base))
    for i in range(num):
        value.extend(base)
    residual = length - len(value)
    value.extend(base[:residual])
    value = np.array(value)
    noise = np.random.normal(0, 1, length)
    value = coef * value + noise_amp * noise
    return value


class UnivariateWaveGenerator(Generator):
    """Generates a univariate signal and append anomalies as desired.


    Attributes:
        train: numpy array of generated train data
        test: numpy array of generated test data
        label: numpy array of labels

    Example:
    ```
        generator = UnivariateWaveGenerator(1000)
        generator.point_global_outliers(0.05, 2, 50)
        train, test, label = generator.get_dataset()
    ```
    """

    _BEHAVIOR_CONFIG = {"freq": 0.04, "coef": 1.5, "offset": 0.0, "noise_amp": 0.05}

    def __init__(
        self,
        stream_length: int,
        train_ratio: float = 0.2,
        behavior=_sine,
        behavior_config: dict = _BEHAVIOR_CONFIG,
    ):
        """
        Args:
            stream_length (int): Total length of `train` and `test` combined
            train_ratio (float, optional): Length of train relative to `stream_length`. Defaults to 0.2.
            behavior (_type_, optional): Underlying signal generating function. Defaults to sine.
            behavior_config (dict, optional): Default config for . Defaults to BEHAVIOR_CONFIG.
        """
        self.STREAM_LENGTH = stream_length
        self.train_ratio = train_ratio
        self.split = int(self.STREAM_LENGTH * self.train_ratio)

        self.behavior = behavior
        self.behavior_config = behavior_config if behavior_config is not None else {}

        self.train = None
        self.data = None
        self.label = None

        self.data_origin = None
        self.timestamp = np.arange(self.STREAM_LENGTH)

        self._generate_timeseries()

    def _generate_timeseries(self):
        self.behavior_config["length"] = self.STREAM_LENGTH
        self.data = self.behavior(**self.behavior_config)

        self.train = self.data[: self.split]
        self.test = self.data[self.split :]

        self.TEST_LENGTH = len(self.test)

        self.test_orig = self.test.copy()
        self.label = np.zeros(self.TEST_LENGTH, dtype=int)

    def point_global_outliers(self, ratio, factor, radius):
        """Add point global outliers to original data

        Args:
            ratio: what ratio outliers will be added
            factor: the larger, the outliers are farther from inliers
            radius: the radius of collective outliers range
        """
        position = (
            np.random.rand(round(self.TEST_LENGTH * ratio)) * self.TEST_LENGTH
        ).astype(int)
        maximum, minimum = max(self.test), min(self.test)
        for i in position:
            local_std = self.test_orig[
                max(0, i - radius) : min(i + radius, self.TEST_LENGTH)
            ].std()
            self.test[i] = self.test_orig[i] * factor * local_std
            if 0 <= self.test[i] < maximum:
                self.test[i] = maximum
            if 0 > self.test[i] > minimum:
                self.test[i] = minimum
            self.label[i] = 1

    def point_contextual_outliers(self, ratio, factor, radius):
        """Add point contextual outliers to original data

        Args:
            ratio: what ratio outliers will be added
            factor: the larger, the outliers are farther from inliers
                    Notice: point contextual outliers will not exceed the range of [min, max] of original data
            radius: the radius of collective outliers range
        """
        position = (
            np.random.rand(round(self.TEST_LENGTH * ratio)) * self.TEST_LENGTH
        ).astype(int)
        maximum, minimum = max(self.test), min(self.test)
        for i in position:
            local_std = self.test[
                max(0, i - radius) : min(i + radius, self.TEST_LENGTH)
            ].std()
            self.test[i] = self.test[i] * factor * local_std
            if self.test[i] > maximum:
                self.test[i] = maximum * min(
                    0.95, abs(np.random.normal(0, 0.5))
                )  # previous(0, 1)
            if self.test[i] < minimum:
                self.test[i] = minimum * min(0.95, abs(np.random.normal(0, 0.5)))

            self.label[i] = 1

    def collective_shapelet_outliers(
        self,
        ratio,
        radius,
        option="square",
        coef=3.0,
        noise_amp=0.0,
        level=5,
        freq=0.04,
        offset=0.0,  # only used when option=='square'
        base=[
            0.0,
        ],
    ):  # only used when option=='other'
        """Add collective global outliers to original data

        Args:
            ratio: what ratio outliers will be added
            radius: the radius of collective outliers range
            option: if 'square': 'level' 'freq' and 'offset' are used to generate square sine wave
                    if 'other': 'base' is used to generate outlier shape
            level: how many sine waves will square_wave synthesis
            base: a list of values that we want to substitute inliers when we generate outliers
        """
        position = (
            np.random.rand(round(self.TEST_LENGTH * ratio / (2 * radius)))
            * self.TEST_LENGTH
        ).astype(int)

        valid_option = {"square", "other"}
        if option not in valid_option:
            raise ValueError("'option' must be one of %r." % valid_option)

        if option == "square":
            sub_data = _square_sine(
                level=level,
                length=self.TEST_LENGTH,
                freq=freq,
                coef=coef,
                offset=offset,
                noise_amp=noise_amp,
            )
        else:
            sub_data = _collective_global_synthetic(
                length=self.TEST_LENGTH, base=base, coef=coef, noise_amp=noise_amp
            )
        for i in position:
            start, end = max(0, i - radius), min(self.TEST_LENGTH, i + radius)
            self.test[start:end] = sub_data[start:end]
            self.label[start:end] = 1

    def collective_trend_outliers(self, ratio, factor, radius):
        """Add collective trend outliers to original data

        Args:
            ratio: what ratio outliers will be added
            factor: how dramatic will the trend be
            radius: the radius of collective outliers range
        """
        position = (
            np.random.rand(round(self.TEST_LENGTH * ratio / (2 * radius)))
            * self.TEST_LENGTH
        ).astype(int)
        for i in position:
            start, end = max(0, i - radius), min(self.TEST_LENGTH, i + radius)
            slope = np.random.choice([-1, 1]) * factor * np.arange(end - start)
            self.test[start:end] = self.test_orig[start:end] + slope
            self.test[end:] = self.test[end:] + slope[-1]
            self.label[start:end] = 1

    def collective_seasonal_outliers(self, ratio, factor, radius):
        """Add collective seasonal outliers to original data

        Args:
            ratio: what ratio outliers will be added
            factor: how many times will frequency multiple
            radius: the radius of collective outliers range
        """
        position = (
            np.random.rand(round(self.TEST_LENGTH * ratio / (2 * radius)))
            * self.TEST_LENGTH
        ).astype(int)
        seasonal_config = self.behavior_config
        seasonal_config["freq"] = factor * self.behavior_config["freq"]
        for i in position:
            start, end = max(0, i - radius), min(self.TEST_LENGTH, i + radius)
            self.test[start:end] = self.behavior(**seasonal_config)[start:end]
            self.label[start:end] = 1

    def get_dataset(self):
        """
        Returns:
            train: numpy array of train data
            test: numpy array of test data
            label: numpy array of labels
        """
        return self.train, self.test, self.label
