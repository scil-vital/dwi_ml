# -*- coding: utf-8 -*-
from dwi_ml.arg_utils import get_memory_args, assert_no_same_args, get_logging_arg
from dwi_ml.models.positional_encoding import (
    keys_to_positional_encodings)
from dwi_ml.models.projects.transformer_models import (
    AbstractTransformerModel)
from dwi_ml.training.utils.batch_loaders import get_args_batch_loader
from dwi_ml.training.utils.batch_samplers import get_args_batch_sampler
from dwi_ml.training.utils.experiment import \
    get_mandatory_args_experiment_and_hdf5, get_comet_args
from dwi_ml.training.utils.trainer import get_training_args

sphere_choices = ['symmetric362', 'symmetric642', 'symmetric724',
                  'repulsion724', 'repulsion100', 'repulsion200']


def get_all_groups_transformers():
    info_embedding = \
        "\n" \
        "  * Note that the input_embedded_size is required for TTST models" \
        " (or nb_cnn_filters if applicable). \n" \
        "  * Total d_model will be input_embedded_size + target_embedded_size.\n" \
        "  * For TTO and TTS models, input embedding size becomes d_model"
    info_target = "\n" \
                  "(FOR MODELS TTO and TTST)"

    (args_input_data, args_input_embedding,
     args_target, args_main_layer) = get_model_arg_groups()
    groups = {
        'Experiment': get_mandatory_args_experiment_and_hdf5(),
        'Batch sampler': get_args_batch_sampler(),
        'Batch loader': get_args_batch_loader(),
        'Trainer': get_training_args(add_a_tracking_validation_phase=True),
        'Memory usage': get_memory_args(add_lazy_options=True, add_rng=True),
        'Comet.ml': get_comet_args(),
        'Transformer: Main data': args_input_data,
        'Transformer: Embedding of the input (X)' + info_embedding:
            args_input_embedding,
        'Transformer: Neighborhood':
            AbstractTransformerModel.get_neighborhood_args_to_parser(),
        'Transformer: Embedding of the target (Y)' + info_target: args_target,
        'Transformer: Main layer params': args_main_layer,
        'Transformer: Output': AbstractTransformerModel.get_args_tracking_model(),
        'Others': get_logging_arg(),
    }

    assert_no_same_args(groups.values(),
                        'IMPLEMENTATION ERROR. Twice the same variable')

    return groups


def get_model_arg_groups():
    args_input_data = {
        'input_group_name': {
            'help': 'Name of the input volume in the hdf5 dataset.'},
        'streamline_group_name': {
            'help': "Name of the streamlines group in the hdf5 dataset."}
    }
    # Step size / compress:
    args_input_data.update(AbstractTransformerModel.get_args_main_model())

    args_input_embedding = AbstractTransformerModel.get_args_input_embedding()
    args_input_embedding['--position_encoding'] = {
        'default': 'sinusoidal', 'metavar': 'k',
        'choices': keys_to_positional_encodings.keys(),
        'help': "Type of positional embedding to use. One of "
                "'sinusoidal' (default)\n or 'relational'. "}

    # TARGETS
    args_target = {
        '--SOS_token_type': {
            'default': 'as_label', 'choices': ['as_label'] + sphere_choices,
            'help':
                "Type of token. SOS is always added in the decoder. "
                "Choices are as_label (a fourth dimension) or \nas class "
                "(directions are sent to classes on the chosen sphere, "
                "and \nan additional class is added for SOS."},
        '--target_embedding_key': {
            'default': 'nn_embedding', 'choices': ['no_embedding', 'nn_embedding'],
            'metavar': 'key',
            'help': "Type of data embedding to use. One of 'no_embedding or "
                    "\n'nn_embedding'."},
        '--target_embedded_size': {
            'type': int, 'metavar': 'n',
            'help': "Embedding size for targets (for TTST only). \n"
                    "Total d_model will be input_embedded_size + "
                    "target_embedded_size."},
    }

    # MAIN LAYER
    args_main_layer = {
        '--model': {
            'choices': {'TTO', 'TTS', 'TTST'}, 'default': 'TTO',
            'help': "One model of transformer amongst the following:\n"
                    " - TTO: Original model. Encoder - Decoder. \n"
                    "   Encoder's input = MRI. Decoder's input = the previous "
                    "direction.\n"
                    " - TTS: Encoder only. Input = MRI (TTS for 'Source').\n"
                    " - TTST: Encoder only. Input and previous direction "
                    "concatenated. (TTST for 'Source + Target'). See [1].\n"
                    "[1]: https://arxiv.org/abs/1905.06596"},
        '--max_len': {
            'type': int, 'default': 1000, 'metavar': 'n',
            'help':
                "Longest sequence allowed. Other sequences will be "
                "zero-padded \nup to that length (but attention can't "
                "attend to padded timepoints).\nPlease beware that this "
                "value influences strongly the executing time and "
                "heaviness.\nAlso used with sinusoidal position "
                "embedding. [1000]"},
        '--nheads': {
            'type': int, 'default': 8, 'metavar': 'n',
            'help':
                "Number of heads per layer. Could be different for each "
                "layer \nbut we decided not to implement this possibility. "
                "[8]"},
        '--n_layers_e': {
            'type': int, 'default': 6, 'metavar': 'n',
            'help': "Number of encoding layers. [6]"},
        '--n_layers_d': {
            'type': int, 'metavar': 'n',
            'help': "TTO MODEL ONLY: Number of decoding layers. "
                    "Default: Same as n_layer_e."},
        '--dropout_rate': {
            'type': float, 'default': 0, 'metavar': 'r',
            'help': "Dropout rate for all dropout layers. Again, could be "
                    "different\n in every layers but that's not the "
                    "choice we made.\n Needed in embedding, encoder and "
                    "decoder. [0]"},
        '--ffnn_hidden_size': {
            'type': int, 'default': None, 'metavar': 'n',
            'help':
                "Size of the feed-forward neural network (FFNN) layer in the "
                "\nencoder and decoder layers. The FFNN is composed of two "
                "linear \nlayers. This is the size of the output of the "
                "first one. \nDefault: d_model/2"},
        '--activation': {
            'choices': ['relu', 'gelu'], 'default': 'relu', 'metavar': 'key',
            'help': "Choice of activation function in the FFNN. One of 'relu' "
                    "or \n'gelu'. [%(default)s]"},
        '--norm_first': {
            'action': 'store_true',
            'help':
                "If set, encoder and decoder layers will perform LayerNorm "
                "before \nother attention and feedforward operations, "
                "otherwise after.\n Torch default + in original paper: False. "
                "\nIn the tensor2tensor code, they suggest that learning is "
                "more robust when \npreprocessing each layer with the norm. "},
        "--start_from_copy_prev": {
            'action': 'store_true',
            'help': "If set, final_output = previous_dir + model_output."}
    }

    return args_input_data, args_input_embedding, args_target, args_main_layer
