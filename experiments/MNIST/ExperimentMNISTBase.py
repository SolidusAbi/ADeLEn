import os
from ..ExperimentBase import ExperimentBase
import config

from ADeLEn.model import ADeLEn

class ExperimentMNISTBase(ExperimentBase):
    def __init__(self) -> None:
        super().__init__()
        self.model = ADeLEn((28, 28), [1, 12, 32], [1024, 512, 128], bottleneck=2)
        self.experiment, self.dataset, self.train_dataset, self.test_dataset = None, None, None, None

    def config(self) -> dict:
        result_dir = os.path.join(config.RESULTS_DIR, 'Chapter8/MNIST/{}'.format(self.experiment))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, )
            os.makedirs(os.path.join(result_dir, 'model'))
            os.makedirs(os.path.join(result_dir, 'imgs'))
            os.makedirs(os.path.join(result_dir, 'result'))

        return {
            'experiment_name': self.experiment,
            'dataset_dir': './data/',
            'model': self.model,
            'batch_size': 128,
            'n_epochs': 50,
            'lr': 1e-3,
            'seed': 42,
            'save_result': True,
            'save_result_dir': os.path.join(result_dir, 'result'),
            'save_imgs_dir': os.path.join(result_dir, 'imgs'),
            'save_model_dir': os.path.join(result_dir, 'model'),
            'save_model_name': 'model.pt',
        }
    
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