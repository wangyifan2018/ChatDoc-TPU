import numpy as np
import time
import os
import sophon.sail as sail


def generate_func(shapes, dtype, mode=1):
    # 0: random
    # 1: zero
    return np.random.random(shapes).astype(dtype) if mode == 0 else np.zeros(shapes).astype(dtype)


class EngineOV:

    def __init__(self, model_path="",batch=4,device_id=0) :
        # 如果环境变量中没有设置device_id，则使用默认值
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])

        self.net = sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.output_names = self.net.get_output_names(self.graph_name)


    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path,self.device_id)


    def __call__(self, input_ids, attention_mask, token_type_ids):
        # print(input_ids)
        # print(attention_mask)
        # print(token_type_ids)
        input_data = {self.input_names[0]: input_ids,
                      self.input_names[1]: attention_mask,
                      self.input_names[2]: token_type_ids}
        outputs = self.net.process(self.graph_name, input_data)
        return  outputs[self.output_names[0]]

