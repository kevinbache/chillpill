"""This is the main training module which is run by the Cloud AI Platform jobs that are launched from
`run_cloud_tuning_job.py`"""
from ritalin import params, callbacks


class MyParams(params.ParameterSet):
    """Define parameters names and default values for development and typechecked autocomplete."""
    # data parameters
    num_samples = 1000
    num_classes = 2
    valid_portion = 0.15
    random_state = 42
    # model parameters
    activation = 'relu'
    num_layers = 2
    num_neurons = 16
    dropout_rate = 0.5
    # training parameters
    learning_rate = 0.01
    batch_size = 16
    num_epochs = 10
    metrics = ['accuracy']


def train_fn(hp: MyParams):
    # generate data
    from sklearn import datasets
    from tensorflow import keras
    x, y = datasets.make_classification(
        n_samples=hp.num_samples,
        random_state=hp.random_state,
        n_classes=hp.num_classes,
    )
    y = keras.utils.to_categorical(y, hp.num_classes)

    # generate model
    inputs = keras.layers.Input(shape=x.shape[1:])
    net = keras.layers.Dense(units=hp.num_neurons, activation=hp.activation)(inputs)
    for _ in range(1, hp.num_layers):
        net = keras.layers.Dense(units=hp.num_neurons, activation=hp.activation)(net)
        if hp.dropout_rate > 0:
            net = keras.layers.Dropout(rate=hp.dropout_rate)(net)

    net = keras.layers.Dense(hp.num_classes, activation='softmax')(net)
    model = keras.models.Model(inputs=inputs, outputs=net)

    model.compile(
        optimizer=keras.optimizers.Adadelta(lr=hp.learning_rate),
        loss=keras.losses.categorical_crossentropy,
        metrics=hp.metrics,
    )

    # train model
    history = model.fit(
        x, y,
        batch_size=hp.batch_size,
        validation_split=hp.valid_portion,
        epochs=hp.num_epochs,
        verbose=2,
        callbacks=[callbacks.GoogleCloudAiCallback()]
    )

    return history


if __name__ == '__main__':
    from ritalin import simple_argparse
    from ritalin import params

    # get a parameter dictionary
    hp = MyParams.from_dict(simple_argparse.args_2_dict())
    print(hp)

    # The important reporting is handled by CmleCallback
    _ = train_fn(hp)
