from enum import Enum

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
    def create_experiment(self, experiment: ExperimentType, reg_factor=1):
        if experiment == ExperimentType.Anomalies_001:
            pass
        else:
            raise NotImplementedError(f'Experiment {experiment} not implemented.')