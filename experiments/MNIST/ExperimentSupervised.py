from .ExperimentMNISTBase import ExperimentMNISTBase
from ..utils.Supervised import SupervisedModel, train

class ExperimentSupervised(ExperimentMNISTBase):
    def __init__(self, known_anomalies, pollution, seed=42) -> None:
        self.model = SupervisedModel((28, 28), [1, 32, 48], [1024, 256, 32])
        self.anomalies_percent = known_anomalies
        self.experiment = 'Supervised/mnist_anomalies_{}_pollution_{}'.format(known_anomalies, pollution)

        super().__init__(known_anomalies, pollution, seed)

    def config(self) -> dict:
        base_config = super().config()
        base_config.update({
            'batch_size': 128,
            'n_epochs': 50,
            'lr': 1e-3,
            'weighted_sampler': False,
            'save_model_name': 'model.pt'
        })

        return base_config
    
    def run(self) -> None:
        self.model = train(**self.config())

        _, _, auc = self.roc_curve()
        return auc