from tensorflow import keras
from hypertune import hypertune


class GoogleCloudAiCallback(keras.callbacks.Callback):
    """A callback that's used to report training results to the Cloud AI for display in the AI Platform > Jobs view"""
    def __init__(self):
        super().__init__()
        self._num_batches_seen = 0
        self._num_samples_seen = 0
        self._hypertune = hypertune.HyperTune()
        self._num_params = None
        self._metrics_dict = {}

    def on_train_begin(self, logs=None):
        self._num_params = self.model.count_params()

    def on_train_batch_begin(self, batch, logs=None):
        self._num_batches_seen += 1
        self._num_samples_seen += logs['size']

    def on_epoch_end(self, epoch, logs=None):
        for k in logs:
            self._metrics_dict[k] = (logs[k], self._num_batches_seen)

    def on_train_end(self, logs=None):
        for k, (val, num_batches) in self._metrics_dict.items():
            self._hypertune.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag=k,
                metric_value=val,
                global_step=num_batches,
            )
