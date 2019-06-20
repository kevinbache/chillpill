import numpy as np

from chillpill import params, tuning


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
        verbose=0,
    )

    return history


if __name__ == '__main__':
    # instantiate the same param class you defined above, overriding some parameters with search ranges
    # the fact that the class is shared
    my_param_ranges = MyParams(
        activation=params.Categorical(['relu', 'tanh']),
        num_layers=params.Integer(min_value=1, max_value=3),
        num_neurons=params.Discrete(np.logspace(2, 8, num=7, base=2)),
        dropout_rate=params.Double(min_value=-0.1, max_value=0.9),
        learning_rate=params.Discrete(np.logspace(-6, 2, 17, base=10)),
        batch_size=params.Integer(min_value=1, max_value=128),
    )

    tuner = tuning.KerasHistoryRandomTuner(
        param_ranges=my_param_ranges,
        num_parameter_sets=10,
        metric_name_of_interest='val_acc'
    )

    tuning.run_tuning(tuner, train_fn)

    best_acc, best_params = tuner.get_best(do_max=True)
    print()
    print(f"best acc:    {best_acc}")
    print(f"best params: {best_params}")
