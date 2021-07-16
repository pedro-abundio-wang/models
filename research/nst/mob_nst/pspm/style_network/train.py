def train_loop():
    """Trains a model using eager + functions.

    This method:
      . Processes the pipeline configs
      . (Optionally) saves the as-run config
      . Builds the model & optimizer
      . Gets the training input data
      . Loads a fine-tuning detection or classification checkpoint if requested
      . Loops over the train data, executing distributed training steps inside tf.functions.
      . Checkpoints the model every `checkpoint_every_n` training steps.
      . Logs the training metrics as TensorBoard summaries.
    """












