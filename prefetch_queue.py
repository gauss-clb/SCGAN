import tensorflow as tf
from datasets import *

slim = tf.contrib.slim


class PrefetchQueue:


    def __init__(self,
                 dataset,
                 preprocessing_fn,
                 batch_size=32,
                 dynamic_pad=False):
        
        '''
            dynamic_pad: Whether the images of batch should have the same size, 
                         if dynamic_pad=True, batch_queue and prefetch_queue will
                         use padding_fifo_queue, otherwise fifo_queue
        '''
        
        self._batch_size = batch_size
        self._tensor_dict = preprocessing_fn(dataset.get())
        self._dynamic_pad = dynamic_pad


        if self._dynamic_pad == True:
            self._static_shape = {key: tensor.get_shape() for key, tensor in self._tensor_dict.items()}
            self._dynamic_shape = {key+'_dynamic_shape': tf.shape(tensor) for key, tensor in self._tensor_dict.items()}
            # pack the dynamic_shape with tensors
            self._tensor_dict.update(self._dynamic_shape)
    

    def get_prefetch_queue(self,                   
                           num_threads=4,
                           capacity=None,
                           prefetch_queue_capacity=4):

        # TODO: allow_smaller_final_batch=True

        _capacity = capacity or 5*self._batch_size
        batch_samples = tf.train.batch(self._tensor_dict,
                                       batch_size=self._batch_size,
                                       num_threads=num_threads,
                                       dynamic_pad=self._dynamic_pad,
                                       capacity=_capacity)

                               
        self._batch_queue = slim.prefetch_queue.prefetch_queue(batch_samples, 
                                                               num_threads=num_threads,
                                                               capacity=prefetch_queue_capacity,
                                                               dynamic_pad=self._dynamic_pad)
    
        return self._batch_queue



    def dequeue(self):

        '''
            If every image of a batch is the same, just return a dict,
            of which items contain value with the shape=[batch_size, ...], 
            otherwise return a list with length of batch_size, every item of
            list is a dict of a sample, like {'image': image_tensor, ...},
        '''
        
        tensor_dict_batch = self._batch_queue.dequeue()
        if not self._dynamic_pad:
            # preprocessing 
            return tensor_dict_batch
        tensor_tuple_dict = {}
        for key in tensor_dict_batch:
            new_key = key
            index = 0
            if '_dynamic_shape' in key:
                new_key = key[:-len('_dynamic_shape')]
                index = 1  
            tensor_tuple = tensor_tuple_dict.setdefault(new_key, [None, None])
            tensor_tuple[index] = tensor_dict_batch[key]  
        
        tensor_dict_list = [{} for i in range(self._batch_size)]
        for key, (tensors, shapes) in tensor_tuple_dict.items():
            for i in range(self._batch_size):
                tensor, shape = tensors[i], shapes[i]
                tensor_dict_list[i][key] = tf.slice(tensor, tf.zeros_like(shape), shape)
                tensor_dict_list[i][key].set_shape(self._static_shape[key])
 
        return tensor_dict_list