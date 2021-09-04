import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import math
from scipy import stats


class SeasonalESD:
    def __init__(self, anomaly_ratio, hybrid, alpha):
        self.anomaly_ratio = anomaly_ratio
        self.hybrid = hybrid
        self.alpha = alpha

        assert self.anomaly_ratio <= 0.49, "anomaly ratio is too high"

        self.n = None
        self.k = None

    def detect(self, data):
        self.n = data.shape[0]
        self.k = math.floor(float(self.n) * self.anomaly_ratio)

        residuals = self._get_residuals(data)
        critical_values = self._calc_critical_values()

        anomaly_indices = self._esd(residuals, critical_values)
        anomalies = np.asarray([])
        if not anomaly_indices.size == 0:
            anomalies = data[anomaly_indices]

        return anomalies, anomaly_indices

    def _get_residuals(self, data):
        median = np.median(data)
        result = self._get_seasonal_decomposition(data)
        residuals = np.asarray(data - result.seasonal - median)
        return residuals

    @staticmethod
    def _get_seasonal_decomposition(data, model='additive', period=7):
        result = seasonal_decompose(data, model=model, period=period)
        return result

    def _esd(self, residuals, critical_values):
        indices, statistics = self._calc_statistics(residuals)

        test_length = len(statistics)
        max_idx = -1
        for i in range(test_length):
            if statistics[i] > critical_values[i]:
                max_idx = i

        anomaly_indices = np.asarray([])
        if max_idx > -1:
            anomaly_indices = np.asarray(indices[: max_idx + 1])

        return anomaly_indices

    def _calc_critical_values(self):
        # critical values are calculates as explained in:
        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm

        critical_values = []
        for i in range(1, self.k+1):
            f_degree = self.n - i - 1
            p = 1 - self.alpha / (2 * (self.n - i + 1))
            t_stat = stats.t.ppf(p, df=f_degree)

            numerator = (self.n - i) * t_stat
            denominator = np.sqrt((self.n - i - 1 + t_stat ** 2) * (self.n - i + 1))
            critical_value = numerator / denominator

            critical_values.append(critical_value)

        return critical_values

    def _calc_statistics(self, data):
        _data = data[:]

        indices = []
        statistics = []
        for i in range(1, self.k + 1):
            idx, statistic = self._calc_statistic(_data)
            statistics.append(statistic)
            indices.append(idx)

            _data = np.delete(_data, idx)

        return indices, statistics

    def _calc_statistic(self, data):
        if self.hybrid:
            median = np.median(data)
            mad = stats.median_absolute_deviation(data)
            statistics = np.asarray(np.abs(data - median) / mad)
        else:
            mean = np.mean(data)
            std = np.std(data)
            statistics = np.asarray(np.abs(data - mean) / std)

        idx = np.argmax(statistics)
        statistic = np.max(statistics)

        return idx, statistic

