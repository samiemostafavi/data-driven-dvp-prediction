import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds

def parquet_tf_pipeline(
    file_addr,
    feature_names,
    label_name,
    dataset_size,
    train_size,
    batch_size,
    dtype=tf.float32,
):
    # We prefetch with a buffer the same size as the dataset because th dataset
    # is very small and fits into memory.
    dataset = (
        tfio.IODataset.from_parquet(
            filename = file_addr,
        )
        .prefetch(buffer_size=dataset_size)
    )

    def read_parquet(features):
        # features is an OrderedDict

        # prepare empty tensors
        keys = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
        values_dict = {}
        
        for a in features.items():
            # look for the features
            for idx,feature_name in enumerate(feature_names):
                # a is a tuple, first item is the key, second is the tensor
                if a[0].decode("utf-8")==feature_name:
                    values = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
                    values = values.write(values.size(), tf.cast(a[1],dtype=dtype))
                    # important to have the squeeze to get (None,) tensor shape
                    values_dict[feature_name] = tf.squeeze(values.stack(),axis=0)

            # look for the keys
            if a[0].decode("utf-8")==label_name:
                keys = keys.write(keys.size(), tf.cast(a[1],dtype=dtype))

        # important to have the squeeze to get (None,) tensor shape
        return (values_dict, tf.squeeze(keys.stack(),axis=0))

    dataset = dataset.map(read_parquet)

    # We shuffle with a buffer the same size as the dataset.
    train_dataset = (
        dataset.take(train_size).cache().shuffle(buffer_size=train_size).batch(batch_size)
    )
    test_dataset = dataset.skip(train_size).take(dataset_size-train_size).cache().shuffle(buffer_size=train_size).batch(batch_size)

    # to check what is being read:
    #for ds in train_dataset:
    #    print(tfds.as_numpy(ds))
    #for ds in test_dataset:
    #    print(tfds.as_numpy(ds))

    return train_dataset, test_dataset

def parquet_tf_pipeline_2(
    file_addr,
    feature_names,
    label_name,
    dataset_size,
    train_size,
    batch_size,
    dtype=tf.float32,
):
    # We prefetch with a buffer the same size as the dataset because th dataset
    # is very small and fits into memory.
    dataset = (
        tfio.IODataset.from_parquet(
            filename = file_addr,
        )
        .prefetch(buffer_size=dataset_size)
    )

    def read_parquet(features):
        # features is an OrderedDict

        # prepare empty tensors
        keys = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
        values_dict = {}
        
        for a in features.items():
            # look for the features
            for idx,feature_name in enumerate(feature_names):
                # a is a tuple, first item is the key, second is the tensor
                if a[0].decode("utf-8")==feature_name:
                    values = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
                    values = values.write(values.size(), tf.cast(a[1],dtype=dtype))
                    # important to have the squeeze to get (None,) tensor shape
                    values_dict[feature_name] = tf.squeeze(values.stack(),axis=0)

            # look for the keys
            if a[0].decode("utf-8")==label_name:
                keys = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
                keys = keys.write(keys.size(), tf.cast(a[1],dtype=dtype))
                values_dict['y_input'] = tf.squeeze(keys.stack(),axis=0)

        # important to have the squeeze to get (None,) tensor shape
        return (values_dict, tf.squeeze(keys.stack(),axis=0))

    dataset = dataset.map(read_parquet)

    # We shuffle with a buffer the same size as the dataset.
    train_dataset = (
        dataset.take(train_size).cache().shuffle(buffer_size=train_size).batch(batch_size)
    )
    test_dataset = dataset.skip(train_size).take(dataset_size-train_size).cache().shuffle(buffer_size=train_size).batch(batch_size)

    # to check what is being read:
    #for ds in train_dataset:
    #    print(tfds.as_numpy(ds))
    #for ds in test_dataset:
    #    print(tfds.as_numpy(ds))

    return train_dataset, test_dataset


def parquet_tf_pipeline_unconditional_single_file(
    file_addr,
    dummy_feature_name,
    label_name,
    dataset_size,
    train_size,
    batch_size,
    dtype=tf.float32,
):

    def read_parquet(features):
        # features is an OrderedDict

        values_dict = {}

        # prepare empty keys tensor
        keys = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
        
        for a in features.items():
            # look for the keys
            if a[0].decode("utf-8")==label_name:
                # found the key, now fill up the values_dict for it
                keys = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
                keys = keys.write(keys.size(), tf.cast(a[1],dtype=dtype))
                values_dict['y_input'] = tf.squeeze(keys.stack(),axis=0)

                # prepare the dummy feature here too, since we have a[1] that 
                # we can create a copy of it with the same shape but all zero
                values = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
                values = values.write(values.size(), tf.cast(tf.zeros_like(a[1]),dtype=dtype))
                # important to have the squeeze to get (None,) tensor shape
                values_dict[dummy_feature_name] = tf.squeeze(values.stack(),axis=0)

        # important to have the squeeze to get (None,) tensor shape
        return (values_dict, tf.squeeze(keys.stack(),axis=0))

    dataset = (
        tfio.IODataset.from_parquet(
            filename = file_addr,
        )
        .prefetch(buffer_size=dataset_size)
    )
    dataset = dataset.map(read_parquet)

    # We shuffle with a buffer the same size as the dataset.
    train_dataset = (
        dataset.take(train_size) \
            #.cache() \
            #.shuffle(buffer_size=train_size) \
            .batch(batch_size)
    )

    #test_dataset = (
    #    dataset.skip(train_size) \
    #        .take(dataset_size-train_size) \
            #.cache() \
            #.shuffle(buffer_size=dataset_size-train_size) \
    #        .batch(batch_size)
    #)
    test_dataset = None

    # to check what is being read:
    print(train_dataset)
    print(f"Train size: {train_size}")
    print("Train dataset:")
    i = 0
    for ds in train_dataset:
        i = i + 1
        print(i)
        print(ds)
        #print(tfds.as_numpy(ds))
    
    #print(f"Dataset size: {dataset_size}")
    #print("Test dataset:")
    #for ds in test_dataset:
    #    print(ds)
        #print(tfds.as_numpy(ds))

    return train_dataset, test_dataset


def parquet_tf_pipeline_unconditional_mutiple_file(
    file_addrs,
    dummy_feature_name,
    label_name,
    dataset_size,
    train_size,
    batch_size,
    dtype=tf.float32,
):

    #def map_fn(file_location):
    #    columns = {label_name : tf.TensorSpec(tf.TensorShape([None,]), tf.double)}
    #
    #    return tfio.IODataset.from_parquet(file_location,columns=columns)

    def read_parquet(features):
        # features is an OrderedDict

        values_dict = {}

        # prepare empty keys tensor
        keys = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)

        for a in features.items():
            print(a)
            # look for the keys
            #if a[0].decode("utf-8")==label_name:
            if a[0].decode("utf-8")==label_name:
                # found the key, now fill up the values_dict for it
                keys = keys.write(keys.size(), tf.cast(a[1],dtype=dtype))

                # prepare the dummy feature here too, since we have a[1] that 
                # we can create a copy of it with the same shape but all zero
                values = values.write(values.size(), tf.cast(tf.zeros_like(a[1]),dtype=dtype))
                

        # important to have the squeeze to get (None,) tensor shape
        # important to insert y_input first
        values_dict['y_input'] = tf.squeeze(keys.stack(),axis=0)
        values_dict[dummy_feature_name] = tf.squeeze(values.stack(),axis=0)
        
        # important to have the squeeze to get (None,) tensor shape
        return (values_dict, tf.squeeze(keys.stack(),axis=0))

    def combine_two_ds(features1,features2):
        # features is an OrderedDict

        #print(features1)
        for a in features1.items():
            if a[0].decode("utf-8")==label_name:
                ten1 = a[1]

        #print(features2)
        for a in features2.items():
            if a[0].decode("utf-8")==label_name:
                ten2 = a[1]

        values1_dict = {}
        values1 = tf.stack([ten1,ten2],axis=0)
        values1_dict[label_name.encode()] = tf.squeeze(values1)

        #values1_dict = {}
        #values1 = tf.concat([features1[label_name],features2[label_name]],axis=0)
        #values1_dict['service_delay'] = tf.squeeze(values1)

        return values1_dict

    def recursive_concat(file_ds):

        if len(file_ds) == 1:
            return file_ds[0]

        new_file_ds = []
        for idx,ds in enumerate(file_ds):

            # combine files
            if idx % 2 == 0:
                dataset = ds
            else:
                # create (2,) from two input datasets
                dataset = tf.data.Dataset.zip((dataset,ds))
                dataset = dataset.map(combine_two_ds)
                # unbatch from (2,) to ()
                dataset = dataset.unbatch()

                new_file_ds.append(dataset)
        
        return recursive_concat(new_file_ds)

    # read files
    #ds = tf.data.Dataset.list_files(file_addrs).map(map_fn)
    file_ds = []
    for idx,file in enumerate(file_addrs):
        file_ds.append((
                tfio.IODataset.from_parquet(
                    filename = file,
                )
                #.prefetch(buffer_size=dataset_size)
            )
        )

    dataset = recursive_concat(file_ds)
    
    # fix structure
    dataset = dataset.map(read_parquet)

    train_dataset = (
        dataset.take(train_size) \
            #.cache() \
            #.shuffle(buffer_size=train_size) \
            .batch(batch_size)
    )

    #test_dataset = (
    #    dataset.skip(train_size) \
    #        .take(dataset_size-train_size) \
            #.cache() \
            #.shuffle(buffer_size=dataset_size-train_size) \
    #        .batch(batch_size)
    #)

    #print(train_dataset)
    # to check what is being read:
    #print(f"Train dataset:")
    #i = 0
    #for ds in train_dataset:
    #    i=i+1
    #    print(i)
    #    print(ds)
        #print(tfds.as_numpy(ds))
    

    return train_dataset

def parquet_tf_pipeline_unconditional_single_file2(
    file_addr,
    dummy_feature_name,
    label_name,
    dataset_size,
    train_size,
    batch_size,
    dtype=tf.float64,
):

    #def map_fn(file_location):
    #    columns = {label_name : tf.TensorSpec(tf.TensorShape([None,]), tf.double)}
    #
    #    return tfio.IODataset.from_parquet(file_location,columns=columns)

    def read_parquet(features):
        # features is an OrderedDict

        values_dict = {}

        # prepare empty keys tensor
        keys = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)

        for a in features.items():
            #print(a)
            # look for the keys
            #if a[0].decode("utf-8")==label_name:
            if a[0].decode("utf-8")==label_name:
                # found the key, now fill up the values_dict for it
                keys = keys.write(keys.size(), tf.cast(a[1],dtype=dtype))

                # prepare the dummy feature here too, since we have a[1] that 
                # we can create a copy of it with the same shape but all zero
                values = values.write(values.size(), tf.cast(tf.zeros_like(a[1]),dtype=dtype))
                

        # important to have the squeeze to get (None,) tensor shape
        # important to insert y_input first
        values_dict['y_input'] = tf.squeeze(keys.stack(),axis=0)
        values_dict[dummy_feature_name] = tf.squeeze(values.stack(),axis=0)
        
        # important to have the squeeze to get (None,) tensor shape
        return (values_dict, tf.squeeze(keys.stack(),axis=0))


    # read files
    #ds = tf.data.Dataset.list_files(file_addrs).map(map_fn)
    dataset = (
        tfio.IODataset.from_parquet(
            filename = file_addr,
        )
        .prefetch(buffer_size=dataset_size)
    )
    
    # fix structure
    dataset = dataset.map(read_parquet)

    train_dataset = (
        dataset \
            .take(train_size) \
    #        .cache() \
    #        .shuffle(buffer_size=train_size) \
            .batch(batch_size)
    )

    #test_dataset = (
    #    dataset.skip(train_size) \
    #        .take(dataset_size-train_size) \
            #.cache() \
            #.shuffle(buffer_size=dataset_size-train_size) \
    #        .batch(batch_size)
    #)

    print(train_dataset)
    # to check what is being read:
    print(f"Train dataset:")
    i = 0
    for ds in train_dataset:
        i=i+1
        print(i)
        print(ds)
        #print(tfds.as_numpy(ds))
    

    return train_dataset

def create_dataset(n_samples = 300, x_dim=3, x_max = 10, x_level=2, dtype = 'float64', dist = 'normal'):

    # generate random sample, two components
    X = np.array(np.random.randint(x_max, size=(n_samples, x_dim))*x_level).astype(dtype)

    if dist is 'normal':
        Y = np.array([ 
                np.random.normal(loc=x_sample[0]+x_sample[1]+x_sample[2],scale=(x_sample[0]+x_sample[1]+x_sample[2])/5)
                    for x_sample in X 
            ]
        ).astype(dtype)
    elif dist is 'gamma':
        Y = np.array([ 
                np.random.gamma(shape=x_sample[0]+x_sample[1]+x_sample[2],scale=(x_sample[0]+x_sample[1]+x_sample[2])/5)
                    for x_sample in X 
            ]
        ).astype(dtype)

    return X,Y


""" load parquet dataset """
def load_parquet(file_addresses, read_columns=None):

    table = pa.concat_tables(
        pq.read_table(
            file_address,columns=read_columns,
        ) for file_address in file_addresses
    )

    return table.to_pandas()
    
