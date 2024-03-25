from enum import Enum

from .ExperimentADeLEn import ExperimentADeLEn

class ExperimentType(Enum):
    Anomalies_001 = 0,
    Anomalies_005 = 1,
    Anomalies_010 = 2,
    Anomalies_020 = 3,
    Pollution_001 = 4,
    Pollution_005 = 5,
    Pollution_010 = 6,
    Pollution_020 = 7

class ExperimentFactory(object):
    @staticmethod
    def create(experiment: ExperimentType, seed=None):
        if experiment == ExperimentType.Anomalies_001:
            return ExperimentADeLEn(0.01, .05, seed)
        if experiment == ExperimentType.Anomalies_005:
            return ExperimentADeLEn(0.05, .05, seed)
        if experiment == ExperimentType.Anomalies_010:
            return ExperimentADeLEn(0.1, .05, seed)
        if experiment == ExperimentType.Anomalies_020:
            return ExperimentADeLEn(0.2, .05, seed)
        else:
            raise NotImplementedError(f'Experiment {experiment} not implemented.')