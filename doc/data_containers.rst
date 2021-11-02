
Understanding our data containers
=================================

Here is how our data is organized to allow torch to use them through Dataloaders. All of the following can be used as lazy instead. Then, the data is only loaded when needed.


- **(Lazy)MultisubjectDataset**

    - subcontainers:

        - .training_set: **Multisubjectsubset**
        - .validation_set: **Multisubjectsubset**
        - .testing_set: **Multisubjectsubset**

    - other attributes:

        - .groups, .volume_groups, .streamline_group, .hdf5_path, .name: general attributes found in the hdf5 file.

    - methods:

        - *.load_data()*: loads the training and validation sets.

- **Multisubjectsubset**

    - subcontainers:

        - .subjects_data_list: **SubjectsDataList** (lazy or not)

    - other attributes:

        - .volume_groups, .streamline_group, .hdf5_path, .set_name: general attributes found in the hdf5 file.
        - .cache_size
        - .streamline_id_slice_per_subj, .total_streamlines, .streamline_length_mm: attributes used by the batch sampler to access specific streamlines.

    - methods:

        - *.get_volume()*: gets a specific mri volume (ID corresponds to the group ID in the config_file) from a specific subject.
        - *.get_volume_verify_cache()*: same, but if data was lazy, checks the volume cache first. If it was not cached, loads it and sends it to the cache.
        - *.__getitem__()*: used by the dataloader. Does not do anything per say, simply returns the sampled streamline id. The batch sampler will do the job of actually loading the data.


- **(Lazy)SubjectsDataList**

    - subcontainers:

        - ._subjects_data_list: List of **SubjectData** (lazy or not)

    - other attributes:

        - .volume_groups, .streamline_group, .hdf5_path: general attributes found in the hdf5 file.
        - .feature_sizes: list of the dimension of each volume.

    - methods:

        - *.add_subject()*: used by the MultisubjectDataset when loading the data.
        - *.__getitem__()*: gets a specific subject from the list. In the lazy case, returns a non-loaded subject.
        - *.getitem_with_handle()*: same, but in the lazy case, adds a hdf5 handle first to allow loading. You probably won't need this method: used in the get_volume() method of the MultisubjectSubset.


- **(Lazy)SubjectData**

    - subcontainers:

        - .volume_list: list of **MRIData** (lazy or not)
        - .sft_data_list: list of **SFTData** (lazy or not)

    - other attributes:
        - .volume_groups, .streamline_group, .subject_id: general attributes found in the hdf5 file.

    - methods:

        - *.init_from_hdf()*: used by the MultisubjectDataset when loadining the data.
        - *.with_handle()*: useful only in the lazy case. Adds hdf_handle to the subject to allow loading.


**MRIData**

    - attributes:

        - ._data: hidden. Depends on the internal data management (lazy or not). Acces the data through the .as_tensor() method.
        - .affine: loaded as a torch tensor even in the lazy case.
        - .shape: property method returning the shape of the data.

    - methods:

        - *.init_from_hdf_info()*: used when loading the data.
        - *.as_tensor()*: gets the data.


**SFTData**

    - attributes:

        - ._space_attributes, .space: properties from the SFT.
        - ._space_attributes_as_tensor: property method converting the space attributes.
        - .streamlines: either an ArraySequence, same as in a SFT, or, in the lazy case, a **LazyStreamlineGetter**, with can access a specific streamline from the hdf5 data without keeping the whole data in memory. It has otherwise the same properties as an ArraySequence.

    - methods:

        - *.init_from_hdf_info()*: used when loadining the data.
        - *.as_tensor()*: gets the data.
