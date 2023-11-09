# -*- coding: utf-8 -*-
from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset


def add_args_testing_subj_hdf5(p, ask_input_group=False,
                               ask_streamlines_group=False):
    p.add_argument('hdf5_file',
                   help="Path to the hdf5 file.")
    p.add_argument('subj_id',
                   help="Subject id to use in the hdf5.")
    if ask_input_group:
        p.add_argument('input_group',
                       help="Model's input's volume group in the hdf5.")
    if ask_streamlines_group:
        p.add_argument('streamlines_group',
                       help="Model's streamlines group in the hdf5.")
    p.add_argument('--subset', default='testing',
                   choices=['training', 'validation', 'testing'],
                   help="Subject id should probably come come the "
                        "'testing' set but you can\n modify this to "
                        "'training' or 'validation'.")


def prepare_dataset_one_subj(hdf5_file, subj_id, lazy=False, cache_size=1,
                             subset_name='testing', volume_groups=None,
                             streamline_groups=None, log_level=None):
    # Right now, we con only track on one subject at the time. We could
    # instantiate a LazySubjectData directly, but we want to use the cache
    # manager (suited better for multiprocessing)
    dataset = MultiSubjectDataset(hdf5_file, lazy=lazy,
                                  cache_size=cache_size,
                                  log_level=log_level)

    possible_subsets = ['training', 'validation', 'testing']
    if subset_name not in possible_subsets:
        raise ValueError("Subset name should be one of {}"
                         .format(possible_subsets))
    load_training = True if subset_name == 'training' else False
    load_validation = True if subset_name == 'validation' else False
    load_testing = True if subset_name == 'testing' else False

    dataset.load_data(load_training, load_validation, load_testing,
                      subj_id, volume_groups, streamline_groups)

    if subset_name == 'testing':
        subset = dataset.testing_set
    elif subset_name == 'training':
        subset = dataset.training_set
    elif subset_name == 'validation':
        subset = dataset.validation_set
    else:
        raise ValueError("Subset must be one of 'training', 'validation' "
                         "or 'testing.")

    if subj_id not in subset.subjects:
        raise ValueError("Subject {} does not belong in hdf5's {} set.\n"
                         "Existing subjects are: \n"
                         "Training set: {}\n"
                         "Validation set: {}\n"
                         "Testing set: {}."
                         .format(subj_id, subset.set_name,
                                 dataset.training_set.subjects,
                                 dataset.validation_set.subjects,
                                 dataset.testing_set.subjects))
    subj_idx = subset.subjects.index(subj_id)  # Should be 0.

    return subset, subj_idx
