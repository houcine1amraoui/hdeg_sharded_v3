import os
import yaml
import torch
from datetime import datetime
from src.utils.get_folders_utils import get_dataset_name

class ModelRegistryManager:
    def __init__(self, config):
        project_root_dir = config["project_root_dir"]
        dataset_name = get_dataset_name(config)
        model_name=config["training"]["model"]

        self.project_root_dir=project_root_dir
        self.dataset_name=dataset_name
        self.model_name=model_name
        self.config = config

        self.exp_dir = os.path.join(self.project_root_dir, "train_experiments", self.model_name, self.dataset_name)
        
        os.makedirs(self.exp_dir, exist_ok=True)

        self.best_model_path = os.path.join(self.exp_dir, "best.pth")
        self.last_model_path = os.path.join(self.exp_dir, "last.pth")

        self.metrics_path = os.path.join(self.exp_dir, "metrics.yaml")
        self.history_path = os.path.join(self.exp_dir, "history.yaml")

        self.best_config_path = os.path.join(self.exp_dir, "best_config.yaml")
        self.last_config_path = os.path.join(self.exp_dir, "last_config.yaml")

        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --------------------------------------------------

    def save_last(self, model, val_loss, epoch):
        torch.save(model.state_dict(), self.last_model_path)

        self._save_yaml(self.last_config_path, self.config)

        self._update_metrics(last_val_loss=val_loss, last_epoch=epoch)

        self._append_history(val_loss, epoch)

    # --------------------------------------------------

    def save_best_if_improved(self, model, val_loss, epoch):

        best_val_loss = self._get_best_val_loss()

        if best_val_loss is None or val_loss < best_val_loss:
            print(f"New best model found: {val_loss:.4f}")
            
            torch.save(model.state_dict(), self.best_model_path)

            self._save_yaml(self.best_config_path, self.config)

            self._update_metrics(
                best_val_loss=val_loss,
                best_epoch=epoch,
                best_date=self.current_time
            )
            
            return True   # improvement happened

        return False      # no improvement

    # --------------------------------------------------

    def _get_best_val_loss(self):

        if not os.path.exists(self.metrics_path):
            return None

        with open(self.metrics_path, "r") as f:
            metrics = yaml.safe_load(f)

        return metrics.get("best_val_loss")

    # --------------------------------------------------

    def _update_metrics(self, **kwargs):

        metrics = {}

        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r") as f:
                metrics = yaml.safe_load(f) or {}

        metrics.update(kwargs)

        self._save_yaml(self.metrics_path, metrics)

    # --------------------------------------------------

    def _append_history(self, val_loss, epoch):

        history = {"experiments": []}

        if os.path.exists(self.history_path):
            with open(self.history_path, "r") as f:
                history = yaml.safe_load(f) or {"experiments": []}

        exp_id = len(history["experiments"]) + 1

        history["experiments"].append(
            {
                "id": exp_id,
                "date": self.current_time,
                "val_loss": float(val_loss),
                "epoch": int(epoch),
                "dataset": self.dataset_name,
                "model": self.model_name
            }
        )

        self._save_yaml(self.history_path, history)

    # --------------------------------------------------

    def _save_yaml(self, path, data):
        with open(path, "w") as f:
            yaml.dump(data, f, sort_keys=False)