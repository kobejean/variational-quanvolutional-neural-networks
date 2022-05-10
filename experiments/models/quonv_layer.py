import logging
from unittest import case
from .scanner import PartialTransposeScanner2D, RandomScanner2D, RasterScanner2D, ZigZagScanner2D

from qiskit import IBMQ

import torch
from torch import nn
import pennylane as qml


#backend = 'ibmq_manila' # 5 qubits
#ibmqx_token = 'XXX'
#IBMQ.save_account(ibmqx_token, overwrite=True)
#IBMQ.load_account()


class QuonvLayer(nn.Module):
    def __init__(self, weights, stride=1, device="default.qubit", wires=4,
                 number_of_filters=1, circuit=None, filter_size=2, out_channels=4, scanner_type = 'raster', seed=None, dtype=torch.float32):

        super(QuonvLayer, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.stride = stride
        self.wires = wires

        # setup device

        if device == "qulacs.simulator":
            self.device = qml.device(device, wires=self.wires, gpu=True)
        elif device == "qulacs.simulator-cpu":
            self.device = qml.device("qulacs.simulator", wires=self.wires, gpu=False)
        elif device == "qiskit.ibmq":
            # IBM quantum computer
            # define your credentials at top of this file
            # and uncomment the IBMQ account saving/loading
            self.device = qml.device('qiskit.ibmq', wires=self.wires, backend=backend)
        else:
            # default simulator
            self.device = qml.device(device, wires=self.wires)


        self.number_of_filters = number_of_filters
        self.filter_size = filter_size
        self.out_channels = out_channels
        self.dtype = dtype


        self.scanner_type = scanner_type
        scanner_gen = {
            'raster': lambda: RasterScanner2D(filter_size=filter_size),
            'random': lambda: RandomScanner2D(filter_size=filter_size),
            'ptrans': lambda: PartialTransposeScanner2D(filter_size=filter_size),
            'zigzag': lambda: ZigZagScanner2D(filter_size=filter_size)
        }
        self.scanner = scanner_gen.get(scanner_type)()
        self.scanner.print()

        self.qlayer = qml.QNode(circuit, self.device, interface="torch", init_method=torch.nn.init.uniform_)
        if weights is not None:
            self.torch_qlayer = qml.qnn.TorchLayer(self.qlayer, weight_shapes={"weights": weights.shape},
                                              init_method=torch.nn.init.uniform_)
            self.torch_qlayer.weights.data = weights
        else:
            self.torch_qlayer = self.qlayer

    def convolve(self, img):
        conv = img.unfold(2, self.filter_size, self.stride)
        conv = conv.unfold(1, self.filter_size, self.stride)
        conv = conv.transpose(2,3).reshape(-1, self.filter_size**2)

    def calc_out_dim(self, img):
        bs, h, w, ch = img.size()
        h_out = (int(h) - self.filter_size) // self.stride + 1
        w_out = (int(w) - self.filter_size) // self.stride + 1
        return bs, h_out, w_out, self.out_channels

    def forward(self, img):
        bs, oh, ow, och = self.calc_out_dim(img)
        out = self.convolve(img)     # b*oh*ow,filter_size**2
        out = self.scanner.scan(out) # b*oh*ow,filter_size**2
        out = self.torch_qlayer(out) # b*oh*ow,och
        out = out.reshape(bs, oh, ow, och)
        return out

    def get_out_template(self, img):
        h, w = img.size()
        h_out = (h - self.filter_size) // self.stride + 1
        w_out = (w - self.filter_size) // self.stride + 1
        return torch.zeros(h_out, w_out)


class ExtractStatesQuonvLayer(QuonvLayer):

    def __init__(self, weights, stride=1, device="default.qubit", wires=4,
                 number_of_filters=1, circuit=None, filter_size=2, out_channels=4, scanner_type = 'raster', seed=None, dtype=torch.complex64):
        super().__init__(weights, stride, device, wires, number_of_filters, circuit, filter_size, out_channels, scanner_type, seed, dtype)

    def calc_out_dim(self, img):
        bs, h, w, ch = img.size()
        h_out = (int(h) - self.filter_size) // self.stride + 1
        w_out = (int(w) - self.filter_size) // self.stride + 1
        return bs, h_out, w_out, 2**self.wires


"""class PreEncodedInputQuonvLayer(QuonvLayer):

    def calc_out_dim(self, img):
        return img.size

    def forward(self, img):

        out = torch.empty(self.calc_out_dim())

        for qnode_inputs, b, j, k in self.convolve(img):

            q_results = self.torch_qlayer(
                qnode_inputs
            )

            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            out[b, j // self.stride, k // self.stride] = q_results"""