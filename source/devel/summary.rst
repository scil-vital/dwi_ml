===================
Summary of our code
===================

1. Creating the hdf5: see dedicated doc page.

2. The trainer abstract :

    - Uses a model-dependent child implementation of the **BatchSamplerAbstract**:

        - For instance, the BatchSequenceSampler creates batches of streamline ids that will be used for each batch iteratively through *__iter__*. Those chosen streamlines can then be loaded and processed with *load_batch*, which also uses data augmentation.

        - For instance, the BatchSequencesSamplerOneInputVolume then uses the generated streamlines to load one underlying input for each timestep of each streamline.

        - The BatchSampler uses the **MultiSubjectDataset** (or lazy)

            - Creates a list of subjects. Using *self.load_data()*, it loops on all subjects (for either the training set or the validation set) and loads the data from the hdf5 (lazily or not).

            - The list is a **DataListForTorch** (or lazy). It contains the subjects but also common parameters such as the feature sizes of each input.

            - The elements are **SubjectData** (or lazy)

                - They contain both the volumes and the streamlines, with the subject ID.

                - Volumes are **MRIData** (or lazy). They contain the data and affine.

                - Streamlines are **SFTData** (or lazy). They contain all information necessary to create a Stateful Tractogram from the streamlines. In the lazy version, streamlines are not loaded all together but read when needed from the **LazyStreamlinesGetter**.
