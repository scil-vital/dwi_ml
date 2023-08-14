# -*- coding: utf-8 -*-
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings
from dwi_ml.models.projects.positional_encoding import (
    keys_to_positional_encodings)
from dwi_ml.models.projects.transforming_tractography import (
    AbstractTransformerModel)
from dwi_ml.models.utils.direction_getters import check_args_direction_getter

sphere_choices = ['symmetric362', 'symmetric642', 'symmetric724',
                  'repulsion724', 'repulsion100', 'repulsion200']


def add_transformers_model_args(p):
    """ Parameters for TransformingTractography"""
    # Mandatory args
    p.add_argument(
        'input_group_name',
        help='Name of the input volume in the hdf5 dataset.')
    p.add_argument(
        'streamline_group_name',
        help="Name of the streamlines group in the hdf5 dataset.")

    # Optional args

    # Step_size / compress
    AbstractTransformerModel.add_args_main_model(p)

    gx = p.add_argument_group("Embedding of the input (X)")
    gx.add_argument(
        '--embedding_key_x', default='nn_embedding',
        choices=keys_to_embeddings.keys(), metavar='key',
        help="Type of data embedding to use. One of 'no_embedding', \n"
             "'nn_embedding' (default) or 'cnn_embedding'.")
    gx.add_argument(
        '--embedding_size_x', type=int, metavar='n',
        help="REQUIRED FOR TTST MODELS: \n"
             "(For TTO and TTS models, embedding size is d_model)"
             "Embedding size for x. Total d_model will be \n"
             "embedding_size_x + embedding_size_t.")
    gx.add_argument(
        '--position_encoding', default='sinusoidal', metavar='key',
        choices=keys_to_positional_encodings.keys(),
        help="Type of positional embedding to use. One of 'sinusoidal' "
             "(default)\n or 'relational'. ")

    gt = p.add_argument_group("Embedding of the target (Y)\n"
                              "(FOR MODELS TTO and TTST)")
    gt.add_argument(
        '--SOS_token_type', default='as_label',
        choices=['as_label'] + sphere_choices,
        help="Type of token. SOS is always added in the decoder. "
             "Choices are as_label (a "
             "fourth dimension) or \nas class (directions are sent to classes "
             "on the chosen sphere, and \nan additional class is added for "
             "SOS.")
    gt.add_argument(
        '--target_embedding', default='nn_embedding',
        choices=keys_to_embeddings.keys(), metavar='key',
        help="Type of data embedding to use. One of 'no_embedding', \n"
             "'nn_embedding' (default) or 'cnn_embedding'.")
    gt.add_argument(
        '--embedding_size_t', type=int, metavar='n',
        help="REQUIRED FOR TTST MODEL: Embedding size for t. \n"
             "Total d_model will be embedding_size_x + embedding_size_t.")

    gtt = p.add_argument_group(title='Transformer: main layers')
    gtt.add_argument(
        '--model', choices={'TTO', 'TTS', 'TTST'}, default='TTO',
        help="One model of transformer amongst the following:\n"
             " - TTO: Original model. Encoder - Decoder. \n"
             "   Encoder's input = MRI. Decoder's input = the previous "
             "direction.\n"
             " - TTS: Encoder only. Input = MRI (TTS for 'Source').\n"
             " - TTST: Encoder only. Input and previous direction "
             "concatenated. (TTST for 'Source + Target'). See [1].\n"
             "[1]: https://arxiv.org/abs/1905.06596")
    gtt.add_argument(
        '--d_model', type=int, metavar='n',
        help="REQUIRED FOR TTO AND TTS MODELS:\n"
             "Output size that will kept constant in all layers to allow skip "
             "connections\n (embedding size, ffnn output size, attention size)."
             "\nDefault in Vaswani: 4096.")
    gtt.add_argument(
        '--max_len', type=int, default=1000, metavar='n',
        help="Longest sequence allowed. Other sequences will be zero-padded \n"
             "up to that length (but attention can't attend to padded "
             "timepoints).\nPlease beware that this value influences strongly "
             "the executing time and heaviness.\nAlso used with sinusoidal "
             "position embedding. [%(default)s]")
    gtt.add_argument(
        '--nheads', type=int, default=8, metavar='n',
        help="Number of heads per layer. Could be different for each layer \n"
             "but we decided not to implement this possibility. [%(default)s]")
    gtt.add_argument(
        '--n_layers_e', type=int, default=6, metavar='n',
        help="Number of encoding layers. [%(default)s]")
    gtt.add_argument(
        '--n_layers_d', type=int, metavar='n',
        help="TTO MODEL ONLY: Number of decoding layers. Default: Same as "
             "n_layer_e.")
    gtt.add_argument(
        '--dropout_rate', type=float, default=0.1, metavar='r',
        help="Dropout rate for all dropout layers. Again, could be different\n"
             "in every layers but that's not the choice we made.\n"
             "Needed in embedding, encoder and decoder. [%(default)s]")
    gtt.add_argument(
        '--ffnn_hidden_size', type=int, default=None, metavar='n',
        help="Size of the feed-forward neural network (FFNN) layer in the \n"
             "encoder and decoder layers. The FFNN is composed of two linear\n"
             "layers. This is the size of the output of the first one. \n"
             "Default: data_embedding_size/2")
    gtt.add_argument(
        '--activation', choices=['relu', 'gelu'], default='relu',
        metavar='key',
        help="Choice of activation function in the FFNN. One of 'relu' or \n"
             "'gelu'. [%(default)s]")
    gtt.add_argument(
        '--norm_first', type=bool, default=False, metavar='True/False',
        help="If True, encoder and decoder layers will perform LayerNorm "
             "before \nother attention and feedforward operations, otherwise "
             "after.\n Torch default + in original paper: False. \nIn the "
             "tensor2tensor code, they suggest that learning is more robust "
             "when \npreprocessing each layer with the norm. Default: False.")
    gtt.add_argument(
        "--start_from_copy_prev", action='store_true',
        help="If set, final_output = previous_dir + model_output.")

    g = p.add_argument_group("Neighborhood")
    AbstractTransformerModel.add_neighborhood_args_to_parser(g)

    g = p.add_argument_group("Output")
    AbstractTransformerModel.add_args_tracking_model(g)
