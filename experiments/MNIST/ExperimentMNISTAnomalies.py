import os
from .ExperimentMNISTBase import ExperimentMNISTBase

class ExperimentMNISTAnomalies(ExperimentMNISTBase):
    def __init__(self, anomalies_percent) -> None:
        super().__init__()
        self.anomalies_percent = anomalies_percent
        self.experiment = 'mnist_anomalies_{}'.format(anomalies_percent)

    def config(self) -> dict:
        base_config = super().config()
        if self.experiment is None:
            return base_config
        
        # base_config['save_result_dir'] = os.path.join(base_config['save_result_dir'], self.experiment)
        # print(base_config['save_result_dir'])
        # print(os.path.exists(base_config['save_result_dir']))
        # if not os.path.exists(base_config['save_result_dir']):
        #     os.makedirs(base_config['save_result_dir'])
        #     os.makedirs(os.path.join(base_config['save_result_dir'], 'model'))
        #     os.makedirs(os.path.join(base_config['save_result_dir'], 'imgs'))
        #     os.makedirs(os.path.join(base_config['save_result_dir'], 'result'))

        return base_config
    
    def run(self):
        raise NotImplementedError

    def save_config(self) -> None:
        config = self.config()
        with open(os.path.join(config['save_result_dir'], 'config.txt'), 'w') as f:
            for k, v in config.items():
                f.write(f'{k}: {v}\n')

    
    def save_model(self, model, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        return self.model.load_model(path)