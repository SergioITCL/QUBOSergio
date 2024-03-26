from functools import partial
from pathlib import Path
from time import time
import numpy as np
import tensorflow as tf
from itcl_inference_engine.network.sequential import Network
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt



from itcl_quantizer.config.models.keras import QuantizerCfg, RoundingQUBOCfg
from itcl_quantizer.config.models.keras import RoundingAnnealerCfg
from itcl_quantizer.tensor_extractor.keras.keras_builder import build

_PARENT = Path(__file__).parent
def main1():
    # load mnist dataset:
    _, (x_test, y_test) = keras.datasets.mnist.load_data()

    # scale mnist
    x_test = x_test.astype("float32") / 255
    x_test = np.array([tf.image.resize(np.expand_dims(image, axis=-1), (28, 28)).numpy() for image in x_test])
    # flatten mnist
    x_test = x_test.reshape(x_test.shape[0], -1)

    data = x_test[:1000]  # first 1000 samples

    def loss(net: Network) -> float:
        res = net.infer(x_test)
        res = res.T.argmax(axis=0)

        hits = 0
        for pred, exp in zip(res, y_test):
            # print(pred, exp)
            if pred == exp:
                hits += 1
        return -hits

    cfg = QuantizerCfg()
    cfg.dense.kernel_dtype = "int4"
    cfg.ada_round_net = RoundingQUBOCfg()  # set the rounding optimizer
    
    t = time()
    net = build(
        model_path=f"{_PARENT}/models/mnist28.h5",
        output_path=f"{_PARENT}/models/quantized.json",
        representative_input=data,  # loss
        loss_fn=loss,
        cfg=cfg,
    )
    print(f"Quantization completed in {time() - t}s")
    t={time() - t}
    seq_net = net.as_sequential_network()
    print("Network Loss:", loss(seq_net))
    return t, loss(seq_net)

def main2():
    # load mnist dataset:
    _, (x_test, y_test) = keras.datasets.mnist.load_data()

    # scale mnist
    x_test = x_test.astype("float32") / 255
    x_test = np.array([tf.image.resize(np.expand_dims(image, axis=-1), (24, 24)).numpy() for image in x_test])
    # flatten mnist
    x_test = x_test.reshape(x_test.shape[0], -1)

    data = x_test[:1000]  # first 1000 samples

    def loss(net: Network) -> float:
        res = net.infer(x_test)
        res = res.T.argmax(axis=0)

        hits = 0
        for pred, exp in zip(res, y_test):
            # print(pred, exp)
            if pred == exp:
                hits += 1
        return -hits

    cfg = QuantizerCfg()
    cfg.dense.kernel_dtype = "int4"
    cfg.ada_round_net = RoundingQUBOCfg()  # set the rounding optimizer
    
    t = time()
    net = build(
        model_path=f"{_PARENT}/models/mnist.h5",
        output_path=f"{_PARENT}/models/quantized.json",
        representative_input=data,  # loss
        loss_fn=loss,
        cfg=cfg,
    )
    print(f"Quantization completed in {time() - t}s")
    t={time() - t}
    seq_net = net.as_sequential_network()
    print("Network Loss:", loss(seq_net))
    return t, loss(seq_net)

if __name__ == "__main__":
    t1, N1= main1()
    with open('resultado.txt', 'w') as f:
        print('Prueba1 28x28 con dos capas', file=f)
        print('Time', t1,'s', file=f)
        print('Network Loss', N1, file=f)
    print('2')
    t2,N2=main1()
    with open('resultado.txt', 'a') as f:
        print('Prueba2 28x28 con dos capas', file=f)
        print('Time', t2,'s', file=f)
        print('Network Loss', N2, file=f)
    print('3')
    t3,N3=main1()
    with open('resultado.txt', 'a') as f:
        print('Prueba3 28x28 con dos capas', file=f)
        print('Time', t3,'s', file=f)
        print('Network Loss', N3, file=f)
    t4, N4= main2()
    with open('resultado.txt', 'a') as f:
        print('Prueba1 24x24 con dos capas', file=f)
        print('Time', t4,'s', file=f)
        print('Network Loss', N4, file=f)
    print('2')
    t5,N5=main2()
    with open('resultado.txt', 'a') as f:
        print('Prueba2 24x24 con dos capas', file=f)
        print('Time', t5,'s', file=f)
        print('Network Loss', N5, file=f)
    print('3')
    t6,N6=main2()
    with open('resultado.txt', 'a') as f:
        print('Prueba3 24x24 con dos capas', file=f)
        print('Time', t6,'s', file=f)
        print('Network Loss', N6, file=f)