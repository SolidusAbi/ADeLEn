from abc import ABC, abstractmethod

def ensure_attributes(*attrs):
    def decorator(cls):
        orig_init = cls.__init__
        def new_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            missing_attrs = [attr for attr in attrs if not hasattr(self, attr)]
            if missing_attrs:
                raise NotImplementedError(f"Subclasses must have the following attributes: {', '.join(missing_attrs)}")
        cls.__init__ = new_init
        return cls
    return decorator

@ensure_attributes('experiment', 'model', 'train_dataset', 'test_dataset')
class ExperimentBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def config(self):
        pass

    @abstractmethod
    def run(self):
        pass
