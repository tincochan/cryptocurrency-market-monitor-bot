#!/usr/bin/python3
"""

Author:Tinc0 CHAN
e-mail: tincochan@foxmail.com

Copyright (C) 2023 CESNET

LICENSE TERMS

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Neither the name of the Company nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

ALTERNATIVELY, provided that this notice is retained in full, this product may be distributed under the terms of the GNU General Public License (GPL) version 2 or later, in which case the provisions of the GPL apply INSTEAD OF those given above.

This software is provided as is'', and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the company or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
"""
import sys
import csv
csv.field_size_limit(sys.maxsize)
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse
from argparse import RawTextHelpFormatter

import math
from scipy.special import gamma

import warnings
warnings.filterwarnings('ignore')

from astropy.timeseries import LombScargle
import astropy

from collections import defaultdict

def fap_significance_test(
    ls: astropy.timeseries.periodograms.lombscargle.core.LombScargle,
    tested_val: np.float64,
    sig_level: float = 0.001,
):
    """Perform the False Alarm Probability library function as significant test.

    Args:
        ls (LombScargle): LombScargle object.
        tested_val (float): Highest power of Lomb-Scargle peridoogram spectrum.
        sig_level (float): Significance level.

    Returns:
        bool: True if FAP is less then sig_level.
    """
    fap = ls.false_alarm_probability(tested_val)
    if fap <= sig_level:
        return True
    return False

def scdf_test(
    power: np.array,
    sig_level: float = 0.1,
    per_level: float = 0.995,
):
    """Perform the SCDF (Scargles Cumulative Distribution Function) significant test.

    Args:
        power (list): List of power of LS periodogram.
        per_level (float, optional): Percent of power for SCDF. Defaults to 0.9.
        sig_level (float, optional): Percent of max power for SCDF. Defaults to 0.01.
        
    Returns:
        tuple: First value is if candidate on periodicity is accepted or denied, and seconde value is confidence.
    """
    s = power.max()
    if s == float("inf"):
        s = sys.float_info.max * sig_level
    else:
        s = s * sig_level
    if 1 - math.exp(-(s) / power.var()) < per_level:
        return False
    return True

def check_constant(X: np.array, constant_level: float = 0.9):
    """Create a histogram and then check if some value doesn't 99 percent of values occur.
    Complexity: O(n)

    Args:
        X (np.array): The values for constant test.
        constant_level (float): The threshold for constant test.

    Returns:
        bool: True if constant (99 percent of one value)
    """
    N = len(X)
    hist = defaultdict(int)
    for i in X:
        hist[i] += 1
    key = max(hist, key=hist.get)
    if hist[key] > constant_level * N:
        return key
    return None

def get_metric_range(metric: list, range_percentage: float = 0.9):
    """Get metric interval/range as metric/feature of time series.

    Args:
        metric (list): Metric of time series (usually packets or bytes).
        range_percentage (float, optional): How many percentage of datapoints must be in computed interval/range. Defaults to 0.9.

    Returns:
        int: Start of interval.
        int: End of interval.
    """
    hist = defaultdict(int)
    s = 0
    for i in metric:
        hist[i] += 1
        s += 1
    keys = list(sorted(hist.keys()))
    threshold = len(metric) * range_percentage
    best_range = None
    best_x = None
    best_y = None
    for i in range(len(keys)):
        x = keys[i]
        sum_vals = 0
        for j in range(i, len(keys)):
            y = keys[j]
            sum_vals += hist[y]
            if sum_vals >= threshold:
                if best_range is None:
                    best_range = y - x
                    best_x = x
                    best_y = y
                elif best_range > y - x:
                    best_x = x
                    best_y = y
    return best_x, best_y

class PeridoicityFeatures:
    def __init__(
            self, 
            id_dependency: str,
            packets: np.ndarray, 
            bytes: np.ndarray, 
            t1: np.ndarray, 
            t2: np.ndarray, 
            power: np.ndarray, 
            frequency: np.ndarray, 
            constant_val: int = None,
            value_singificance: float = 0.2, 
            range_percentage: float = 0.9,
            label: str = "",
        ) -> None:
        self.id_dependency = id_dependency
        self.packet_value, self.packet_value_x, self.packet_value_y = self.metric_features(packets, constant_val=constant_val, value_singificance=value_singificance, range_percentage=range_percentage)
        self.packet_mean, self.packet_std, self.packet_skewness, self.packet_kurtosis = self.metric_statistics(packets)
        self.bytes_value, self.bytes_value_x, self.bytes_value_y = self.metric_features(bytes, constant_val=constant_val, value_singificance=value_singificance, range_percentage=range_percentage)
        self.bytes_mean, self.bytes_std, self.bytes_skewness, self.bytes_kurtosis = self.metric_statistics(bytes)
        duration = t2 - t1
        self.duration_value, self.duration_value_x, self.duration_value_y = self.metric_features(duration, constant_val=constant_val, value_singificance=value_singificance, range_percentage=range_percentage)
        self.duration_mean, self.duration_std, self.duration_skewness, self.duration_kurtosis = self.metric_statistics(duration)
        difftimes = t2[:-1] - t1[1:]
        self.difftimes_value, self.difftimes_value_x, self.difftimes_value_y = self.metric_features(difftimes, constant_val=constant_val, value_singificance=value_singificance, range_percentage=range_percentage)
        self.difftimes_mean, self.difftimes_std, self.difftimes_skewness, self.difftimes_kurtosis = self.metric_statistics(difftimes)
        self.max_power = -1
        self.max_frequency = -1
        self.min_power = -1
        self.min_frequency = -1
        self.spectral_energy = -1
        self.spectral_entropy = -1
        self.spectral_kurtosis = -1
        self.spectral_skewness = -1
        self.spectral_rolloff = -1
        self.spectral_cetroid = -1
        self.spectral_spread = -1
        self.spectral_slope = -1
        self.spectral_crest = -1
        self.spectral_flux = -1
        self.spectral_bandwidth = -1
        if power is not None:
            self.spectral_features(power, frequency)
        self.label = label
    
    def metric_statistics(self,  metric: np.ndarray):
        metric_mean = metric.mean()
        metric_std = metric.std()
        return (
            metric_mean, 
            metric_std, 
            np.sum(np.power(metric - metric_mean, 3)) / np.power(metric_std, 3),
            np.sum(np.power(metric - metric_mean, 4)) / np.power(metric_std, 4),
        )
    
    def metric_features(self, metric: np.ndarray, constant_val: int = None, value_singificance: float = 0.2, range_percentage: float = 0.9):
        metric_value = 0
        metric_value_x = 0
        metric_value_y = 0
        if constant_val is None:
            hist = defaultdict(int)
            for i in metric:
                hist[i] += 1
            s = sum(hist.values())
            key1 = max(hist, key=hist.get)
            hist.pop(key1)
            if len(hist.keys()) == 0:
                metric_value = key1
            else:    
                key2 = max(hist, key=hist.get)
                if (hist[key1] / s) - value_singificance  >  hist[key2] / s:
                    metric_value = key1
                elif abs(hist[key1] / s -   hist[key2] / s) < 0.1:
                    metric_value_x = key1
                    metric_value_y = key2
                else:
                    metric_value_x, metric_value_y = get_metric_range(metric, range_percentage)
        else:
            metric_value = constant_val
        return metric_value, metric_value_x, metric_value_y
    
    def spectral_features(self, power: np.ndarray, frequency: np.ndarray, rolloff_threshold: float = 0.85):
        # compute features
        self.max_power = power.max()
        tmp = np.asarray(power == self.max_power).nonzero()[0]
        if len(tmp) > 0:
            self.max_frequency = frequency[tmp[0]]
        self.min_power = power.min()
        tmp = np.asarray(power == self.min_power).nonzero()[0]
        if len(tmp) > 0:
            self.min_frequency = frequency[tmp[0]]
        # Spectral Energy: The total energy present at all frequencies in the periodogram
        self.spectral_energy = power.sum()
        # Spectral Entropy: A measure of the randomness or disorfer in periodogram
        self.spectral_entropy = -np.sum(power * np.log2(power))
        # Spectral Kurtosis (Flatness): A measure of the uniformity of the power spectrum.
        self.spectral_kurtosis = np.sum(np.power(power - np.mean(power), 4)) / np.power(np.std(power), 4)
        # Spectral Skewness: A measure of the peakedness or flatness of the power spectrum
        self.spectral_skewness = np.sum(np.power(power - np.mean(power), 3)) / np.power(np.std(power), 3)
        # Spectral Rolloff: The frequecny at which the power spectrum falls off significantly
        threshold = rolloff_threshold * self.max_power
        rolloff_idx = np.argmax(power > threshold)
        self.spectral_rolloff =  frequency[rolloff_idx]
        # Spectral Centroid: The average frequency of the power spectrum
        self.spectral_cetroid = np.sum(frequency * power) / np.sum(power)
        # Spectral Spread: The difference between the highest and lowest frequencies in the power spectrum
        self.spectral_spread = self.max_power - self.min_power
        # Spectral Slope: The slope of trend of the power spectrum over a given frequency range
        self.spectral_slope = np.polyfit(np.log(frequency), np.log(power), deg=1)[0]
        # Spectral Crest: the ratio of the peak spectral magnitude to the average spectral magnitude
        self.spectral_crest = self.max_power / np.mean(power)
        # Spectral Flux: the rate of change of the spectral energy over time
        self.spectral_flux = np.sum(np.abs(np.diff(power)))
        # Spectral Bandwidth: the difference between the upper and lower frequencies at which the spectral energy is half of its maximum value
        self.spectral_bandwidth = frequency[np.argmax(power)] - frequency[np.argmin(power)]
    
    def export(self):
        return [
            self.id_dependency,
            self.label,
            self.packet_value, 
            self.packet_value_x, 
            self.packet_value_y,
            self.packet_mean, 
            self.packet_std, 
            self.packet_skewness, 
            self.packet_kurtosis,
            self.bytes_value, 
            self.bytes_value_x, 
            self.bytes_value_y,
            self.bytes_mean, 
            self.bytes_std, 
            self.bytes_skewness, 
            self.bytes_kurtosis,
            self.duration_value, 
            self.duration_value_x, 
            self.duration_value_y,
            self.duration_mean, 
            self.duration_std, 
            self.duration_skewness, 
            self.duration_kurtosis,
            self.difftimes_value, 
            self.difftimes_value_x, 
            self.difftimes_value_y,
            self.difftimes_mean, 
            self.difftimes_std, 
            self.difftimes_skewness, 
            self.difftimes_kurtosis,
            self.max_power,
            self.max_frequency,
            self.min_power,
            self.min_frequency,
            self.spectral_energy,
            self.spectral_entropy,
            self.spectral_kurtosis,
            self.spectral_skewness,
            self.spectral_rolloff,
            self.spectral_cetroid,
            self.spectral_spread,
            self.spectral_slope,
            self.spectral_crest,
            self.spectral_flux,
            self.spectral_bandwidth,
        ]
        
HEADER = [
                "id_dependency",
                "label",
                "packet_value", 
                "packet_value_x", 
                "packet_value_y",
                "packet_mean", 
                "packet_std", 
                "packet_skewness", 
                "packet_kurtosis",
                "bytes_value", 
                "bytes_value_x", 
                "bytes_value_y",
                "bytes_mean", 
                "bytes_std", 
                "bytes_skewness", 
                "bytes_kurtosis",
                "duration_value", 
                "duration_value_x", 
                "duration_value_y",
                "duration_mean", 
                "duration_std", 
                "duration_skewness", 
                "duration_kurtosis",
                "difftimes_value", 
                "difftimes_value_x", 
                "difftimes_value_y",
                "difftimes_mean", 
                "difftimes_std", 
                "difftimes_skewness", 
                "difftimes_kurtosis",
                "max_power",
                "max_frequency",
                "min_power",
                "min_frequency",
                "spectral_energy",
                "spectral_entropy",
                "spectral_kurtosis",
                "spectral_skewness",
                "spectral_rolloff",
                "spectral_cetroid",
                "spectral_spread",
                "spectral_slope",
                "spectral_crest",
                "spectral_flux",
                "spectral_bandwidth",
            ]
        
def get_frequencies(min_delta: float = 2, precision:float = 0.25):
    T = 7400  # duration of time series in seconds
    Pmin = 1  # minimum period in seconds
    Pmax = T/min_delta  # maximum period in seconds
    N = 10000 * precision    # frequency resolution -- This will depend on the desired precision of the periodogram and the amount of 
                # computational resources you have available.
    fmin = 1/Pmax  # minimum frequency in Hz
    fmax = 1/Pmin  # maximum frequency in Hz
    df = (fmax - fmin)/N  # frequency resolution
    return  np.arange(fmin, fmax, df) # create frequencies


def parse_arguments():
    """Function for set arguments of module.

    Returns:
        argparse: Return setted argument of module.
    """
    parser = argparse.ArgumentParser(
        description="""

    Usage:""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--time_series",
        help="CSV file with time series.",
        type=str,
        metavar="NAME.SUFFIX",
        default=None,
    )
    parser.add_argument(
        "--peridoicity_file",
        help="Output CSV file.",
        type=str,
        metavar="NAME.SUFFIX",
        default=None,
    )
    parser.add_argument(
        "--label",
        help="FIX LABEL.",
        type=str,
        metavar="LABEL",
        default=None,
    )
    parser.add_argument(
        "--sig_level",
        help="Significance level of Ls periodogram periodicity test.",
        type=float,
        metavar="FLOAT",
        default=0.1,
    )
    parser.add_argument(
        "--per_level",
        help="Periodicity level of Ls periodogram periodicity test.",
        type=float,
        metavar="FLOAT",
        default=0.99,
    )
    # arg., arg.=
    arg = parser.parse_args()
    return arg


def main():
    """Main function of the module."""
    arg = parse_arguments()

    if arg.peridoicity_file is None:
        periodic_file = f"{arg.time_series.split('.csv')[0]}.periodicity_features.csv"
    else:
        periodic_file = arg.peridoicity_file

    cnt = 0
    cnt_continue = 0
    cnt_per = 0
    cnt_con = 0
    multiple_labels = 0
    print(f"\rAll: {cnt_continue-cnt}, Tested: {cnt}, Periodic: {cnt_per}, Constant: {cnt_con}      Multiple label: {multiple_labels}", end="")
    with open(periodic_file, "w") as w:
        writer = csv.writer(w, delimiter=",")
        writer.writerow(HEADER)
        with open(arg.time_series,"r") as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                id_dependency = row[0]
                number_of_flow = int(row[1])
                cnt_continue += 1
                if number_of_flow < 10:
                    continue
                cnt += 1
                packets = np.array(json.loads(row[4]))
                t1 = np.array(json.loads(row[6]))
                try:
                    ls = LombScargle(t1, packets)
                    # Set frequency range by "hand"
                    # frequency = get_frequencies(min_delta = 2, precision = 0.25)
                    # power = ls.power(frequency)
                    # Set frequency range automaticaly
                    frequency, power = ls.autopower() # minimum_frequency=fmin, maximum_frequency=fmax,samples_per_peak=10)
                    power = power[np.logical_not(np.isnan(power))]
                except:
                    continue
                # Set precision of periodicity detection: 
                #       sig_level - lower value is more strict
                #       per_level - higher value is more strict 
                #       constant_level - higher value is more strict
                if len(power) == 0:
                    constant_val = check_constant(packets[:1000], constant_level = 0.9)
                    if constant_val is None:
                        continue
                    cnt_con += 1
                    bytes = np.array(json.loads(row[5]))
                    t2 = np.array(json.loads(row[7]))
                    if arg.label is not None:
                        _label = arg.label
                    else:
                        labels = list(set(json.loads(row[8].replace("'",'"'))))
                        if len(labels) > 1:
                            multiple_labels += 1
                        _label = labels[0]
                    features = PeridoicityFeatures(id_dependency, packets, bytes, t1, t2, None, None, label=_label)
                    writer.writerow(features.export())    
                    continue
                if scdf_test(abs(power), sig_level=arg.sig_level, per_level=arg.per_level) is False:
                    constant_val = check_constant(packets[:1000], constant_level = 0.9)
                    if constant_val is None:
                        continue
                    cnt_con += 1
                else:
                    cnt_per += 1
                bytes = np.array(json.loads(row[5]))
                t2 = np.array(json.loads(row[7]))
                if arg.label is not None:
                    _label = arg.label
                else:
                    labels = list(set(json.loads(row[8].replace("'",'"'))))
                    if len(labels) > 1:
                        multiple_labels += 1
                    _label = labels[0]
                features = PeridoicityFeatures(id_dependency, packets, bytes, t1, t2, power, frequency, label=_label)
                writer.writerow(features.export())
                print(f"\rAll: {cnt_continue-cnt}, Tested: {cnt}, Periodic: {cnt_per}, Constant: {cnt_con}      Multiple label: {multiple_labels}", end="",flush=False)
    print("")


if __name__ == "__main__":
    main()
