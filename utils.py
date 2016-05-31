from workspace import *


def transpose_stream(data):
    #     data is a tuple, since it's expected to come from the padding transformer
    return tuple(np.swapaxes(item, 0, 1) for item in data)


def default_batch_stream(dataset, batch_size):
    batch_scheme = SequentialScheme(dataset.num_examples, batch_size=batch_size)
    return DataStream.default_stream(dataset, iteration_scheme=batch_scheme)


def fav_extensions(n_epochs, variables_of_interest, save_path, monitor_freq=1000):
    # add monitoring freq
    return [FinishAfter(after_n_epochs=n_epochs),
            TrainingDataMonitoring(variables_of_interest, every_n_batches=monitor_freq),
            Timing(after_epoch=True),
            Printing(on_interrupt=True, every_n_batches=monitor_freq),
            Checkpoint(save_path)
            # Plot("Example Plot", channels=[['test_cost_simple_xentropy', "test_error_rate"]])
            ]


def load_tar(file_path):
    return load(open(file_path, 'rb'))
