import numpy as np
import matplotlib.pyplot as plt

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_USER,MLFLOW_SOURCE_NAME
from omegaconf import DictConfig, ListConfig

class MlflowWriter():
    def __init__(self,experiment_name):
        self.client = MlflowClient()
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except Exception as e:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        self.experiment = self.client.get_experiment(self.experiment_id)
        print("New experiment started")
        print(f"Name: {self.experiment.name}")
        print(f"Experiment_id: {self.experiment.experiment_id}")
        print(f"Artifact Location: {self.experiment.artifact_location}")

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)


    def _explore_recursive(self,parsent_name, element):
        if isinstance(element,DictConfig):
            for k,v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f'{parsent_name}.{k}',v)
        elif isinstance(element,ListConfig):
            for i,v in enumerate(element):
                self.client.log_param(self.run_id, f'{parsent_name}',element)

