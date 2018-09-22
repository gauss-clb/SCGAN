import tensorflow as tf
import os

slim = tf.contrib.slim

_FILE_PATTERN = 'celeba_?'

SPLITS_TO_SIZES = {'train': 201563}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A RGB image with shape (128,128,3)',
    'height': 'The height of image.',
    'width': 'The width of image.',
    'filename': 'The filename of image.'
}


def get_split(dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading celeba.
    Args:
        dataset_dir: The base directory of the dataset sources.
        file_pattern: The file pattern to use when matching the dataset sources.
            It is assumed that the pattern contains a '%s' string so that the split
            name can be inserted.
        reader: The TensorFlow reader type.
    Returns:
        A `Dataset` namedtuple.
    """

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern)
    
    # Allowing None in the signature so that dataset_factory can use the default.
    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {
        'filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'height': tf.FixedLenFeature((), tf.int64, 1),
        'width': tf.FixedLenFeature((), tf.int64, 1),
    }

    items_to_handlers = {
        'filename': slim.tfexample_decoder.Tensor('filename'),
        'image': slim.tfexample_decoder.Image(
            image_key='encoded', format_key='format', shape=[128, 128, 3]),
        'height': slim.tfexample_decoder.Tensor('height'),
        'width': slim.tfexample_decoder.Tensor('width'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES['train'],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=1,
        labels_to_names=None)


'''
You also can set parameters of DatasetDataProvider,
such as num_readers, common_queue_capacity=20*batch_size, common_queue_min=10*batch_size,...
'''

class CelebA(object):

    def __init__(self, 
                 data_dir, 
                 num_readers=1,
                 shuffle=True,
                 num_epochs=None,
                 common_queue_capacity=256,
                 common_queue_min=128):

        self.data_dir = data_dir
        dataset = get_split(data_dir)
        self.provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                                       num_readers=num_readers, 
                                                                       shuffle=shuffle, 
                                                                       num_epochs=num_epochs,
                                                                       common_queue_capacity=common_queue_capacity, 
                                                                       common_queue_min=common_queue_min)
        self.num_samples = dataset.num_samples

    def get(self):
        '''
            filename: The filename of image, such as 000001.jpg.
            image: A RGB image of varying height and width
            height: The height of image
            width: The width of image

            return:
                filename: the filename of image
                image: RGB
                height: the height of image
                width: the width of image

        '''
        filename, image, height, width = self.provider.get(['filename', 'image', 'height', 'width'])
        return {
            'filename': filename,
            'image': image,
            'height': height,
            'width': width
        }