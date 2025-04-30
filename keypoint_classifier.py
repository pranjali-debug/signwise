#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        
        # Create a copy of the input data to avoid reference issues
        input_data = np.array([landmark_list], dtype=np.float32).copy()
        
        self.interpreter.set_tensor(
            input_details_tensor_index,
            input_data)
        
        self.interpreter.invoke()
        
        output_details_tensor_index = self.output_details[0]['index']
        # Make a copy of the output tensor to avoid reference issues
        result = self.interpreter.get_tensor(output_details_tensor_index).copy()
        
        nres = np.squeeze(result)
        # print(nres)
        # at correct gesture it gives 0.7 for b approx
        result_index = np.argmax(nres)
        #TODO: find the appropriate threshold
        if(nres[result_index] < 0.48):  #0.6 , 0.5 , 0.7 try all these
            result_index = 27  # Change -1 to 27 for "not recognised" index
            
        return result_index