"""
ModelHandler defines an example model handler for load and inference requests for MXNet CPU models
"""
import json
import logging
import os
from collections import namedtuple
from io import StringIO

# import mxnet as mx
import mlflow 
import xgboost as xgb
import numpy as np
from sagemaker_inference import (
    content_types,
    decoder,
    default_inference_handler,
    encoder,
    errors,
    utils,
)
import pandas as pd 

logger = logging.getLogger(__name__)


class ModelHandler(object):
    """
    A sample Model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.model = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir")

        print("model_dir {}".format(model_dir))
        print(os.system("ls {}".format(model_dir)))

        self.model = mlflow.xgboost.load_model(model_dir)


    def input_fn(self, input_data, content_type="text/csv"):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready

        if content_type == "text/csv":
            print("input_data {}".format(input_data))
            print("type(input_data) {}".format(type(input_data)))
            df = pd.read_csv(StringIO(input_data), header=None, index_col=False, sep=",")
            return df
        else:
            raise ValueError("{} not supported by script!".format(content_type))


    def predict_fn(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        return self.model.inplace_predict(model_input)


    def output_fn(self, prediction, accept):
        """
        Return predict result in as list.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        return encoder.encode(prediction, accept)

    def handle(self, input_data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """

        model_input = self.input_fn(input_data)
        model_out = self.predict_fn(model_input)
        return self.output_fn(model_out)


_service = ModelHandler()


def handle(input_data, context):
    if not _service.initialized:
        _service.initialize(context)

    if input_data is None:
        return None

    return _service.handle(input_data, context)