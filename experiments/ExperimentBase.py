class ExperimentBase:
    def __init__(self) -> None:
        self.model = None
        self.dataset = None

    def config(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError