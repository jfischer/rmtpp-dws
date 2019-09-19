#!/usr/bin/env python
import click
import tf_rmtpp
import tensorflow as tf
import tempfile

from dataworkspaces.lineage import LineageBuilder

def_opts = tf_rmtpp.rmtpp_core.def_opts

@click.command()
@click.argument('event_train_file')
@click.argument('time_train_file')
@click.argument('event_test_file')
@click.argument('time_test_file')
@click.option('--summary', 'summary_dir', help='Which folder to save summaries to.', default=None)
@click.option('--metrics', 'metrics_dir',
              help='Which folder to save metrics to.', default=None)
@click.option('--epochs', 'num_epochs', help='How many epochs to train for.', default=1)
@click.option('--restart/--no-restart', 'restart', help='Can restart from a saved model from the summary folder, if available.', default=False)
@click.option('--train-eval/--no-train-eval', 'train_eval', help='Should evaluate the model on training data?', default=False)
@click.option('--test-eval/--xno-test-eval', 'test_eval', help='Should evaluate the model on test data?', default=True)
@click.option('--scale', 'scale', help='Constant to scale the time fields by.', default=1.0)
@click.option('--batch-size', 'batch_size', help='Batch size.', default=def_opts.batch_size)
@click.option('--bptt', 'bptt', help='Series dependence depth.', default=def_opts.bptt)
@click.option('--init-learning-rate', 'learning_rate', help='Initial learning rate.', default=def_opts.learning_rate)
@click.option('--cpu-only/--no-cpu-only', 'cpu_only', help='Use only the CPU.', default=def_opts.cpu_only)
def cmd(event_train_file, time_train_file, event_test_file, time_test_file,
        summary_dir, metrics_dir, num_epochs, restart, train_eval, test_eval, scale,
        batch_size, bptt, learning_rate, cpu_only):
    """Read data from EVENT_TRAIN_FILE, TIME_TRAIN_FILE and try to predict the values in EVENT_TEST_FILE, TIME_TEST_FILE."""

    builder = LineageBuilder().as_script_step()\
                .with_parameters({
                    'num_epochs':num_epochs,
                    'scale':scale,
                    'batch_size':batch_size,
                    'bptt':bptt,
                    'learning_rate':learning_rate
                })\
                .with_input_paths(event_train_file, time_train_file,
                                  event_test_file, time_test_file)
    if metrics_dir and test_eval:
        builder.as_results_step(metrics_dir)
    with builder.eval() as lineage:
        if summary_dir:
            lineage.add_output_path(summary_dir)

        data = tf_rmtpp.utils.read_data(
            event_train_file=event_train_file,
            event_test_file=event_test_file,
            time_train_file=time_train_file,
            time_test_file=time_test_file
        )

        data['train_time_out_seq'] /= scale
        data['train_time_in_seq'] /= scale
        data['test_time_out_seq'] /= scale
        data['test_time_in_seq'] /= scale

        tf.reset_default_graph()
        sess = tf.Session()

        tf_rmtpp.utils.data_stats(data)

        rmtpp_mdl = tf_rmtpp.rmtpp_core.RMTPP(
            sess=sess,
            num_categories=data['num_categories'],
            summary_dir=summary_dir if summary_dir is not None else tempfile.mkdtemp(),
            batch_size=batch_size,
            bptt=bptt,
            learning_rate=learning_rate,
            cpu_only=cpu_only,
            _opts=tf_rmtpp.rmtpp_core.def_opts
        )

        # TODO: The finalize here has to be false because tf.global_variables()
        # creates a new graph node (why?). Hence, need to be extra careful while
        # saving the model.
        rmtpp_mdl.initialize(finalize=False)
        rmtpp_mdl.train(training_data=data, restart=restart,
                        with_summaries=summary_dir is not None,
                        num_epochs=num_epochs, with_evals=False)

        if train_eval:
            print('\nEvaluation on training data:')
            train_time_preds, train_event_preds = rmtpp_mdl.predict_train(data=data)
            rmtpp_mdl.eval(train_time_preds, data['train_time_out_seq'],
                           train_event_preds, data['train_event_out_seq'])
            print()

        if test_eval:
            print('\nEvaluation on testing data:')
            test_time_preds, test_event_preds = rmtpp_mdl.predict_test(data=data)
            (mae, total_valid, acc) = rmtpp_mdl.eval(test_time_preds,
                                                     data['test_time_out_seq'],
                                                     test_event_preds,
                                                     data['test_event_out_seq'])
            if metrics_dir:
                lineage.write_results({
                    'MAE':mae,
                    'total_valid':total_valid,
                    'ACC':acc
                })
        print()


if __name__ == '__main__':
    cmd()

