import torch
import argparse
import onnx
import torch.nn as nn
import io
from onnxsim import simplify
import sys
from tqdm import tqdm
from time import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--bs", type=int, default=1)
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()

if args.model == "segformer":
    from transformers import SegformerForSemanticSegmentation
    class SegFormer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        
        def forward(self, x):
            return self.model(x).logits
    model = SegFormer()
    res = 512
elif args.model == "candy":
    if not args.profile:
        import os
        import onnx_graphsurgeon as gs
        os.system("wget https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/validated/vision/style_transfer/fast_neural_style/model/candy-9.onnx --quiet")
        model = onnx.shape_inference.infer_shapes(simplify(onnx.load("candy-9.onnx"))[0])
        graph = gs.import_onnx(model)
        graph.inputs[0].shape[0] = args.bs
        for node in graph.nodes:
            node.outputs[0].shape[0] = args.bs
        model = gs.export_onnx(graph)
        onnx.save(model, f"candy_bs{args.bs}.onnx")
        os.system("rm candy-9.onnx")
        exit()
    else:
        class TransformerNet(torch.nn.Module):
            def __init__(self):
                super(TransformerNet, self).__init__()
                # Initial convolution layers
                self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
                self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
                self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
                self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
                self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
                self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
                # Residual layers
                self.res1 = ResidualBlock(128)
                self.res2 = ResidualBlock(128)
                self.res3 = ResidualBlock(128)
                self.res4 = ResidualBlock(128)
                self.res5 = ResidualBlock(128)
                # Upsampling Layers
                self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
                self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
                self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
                self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
                self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
                # Non-linearities
                self.relu = torch.nn.ReLU()

            def forward(self, X):
                y = self.relu(self.in1(self.conv1(X)))
                y = self.relu(self.in2(self.conv2(y)))
                y = self.relu(self.in3(self.conv3(y)))
                y = self.res1(y)
                y = self.res2(y)
                y = self.res3(y)
                y = self.res4(y)
                y = self.res5(y)
                y = self.relu(self.in4(self.deconv1(y)))
                y = self.relu(self.in5(self.deconv2(y)))
                y = self.deconv3(y)
                return y


        class ConvLayer(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride):
                super(ConvLayer, self).__init__()
                reflection_padding = kernel_size // 2
                self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
                self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

            def forward(self, x):
                out = self.reflection_pad(x)
                out = self.conv2d(out)
                return out


        class ResidualBlock(torch.nn.Module):
            """ResidualBlock
            introduced in: https://arxiv.org/abs/1512.03385
            recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
            """

            def __init__(self, channels):
                super(ResidualBlock, self).__init__()
                self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
                self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
                self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
                self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                residual = x
                out = self.relu(self.in1(self.conv1(x)))
                out = self.in2(self.conv2(out))
                out = out + residual
                return out


        class UpsampleConvLayer(torch.nn.Module):
            """UpsampleConvLayer
            Upsamples the input and then does a convolution. This method gives better results
            compared to ConvTranspose2d.
            ref: http://distill.pub/2016/deconv-checkerboard/
            """

            def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
                super(UpsampleConvLayer, self).__init__()
                self.upsample = upsample
                reflection_padding = kernel_size // 2
                self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
                self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

            def forward(self, x):
                x_in = x
                if self.upsample:
                    x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
                out = self.reflection_pad(x_in)
                out = self.conv2d(out)
                return out
        model = TransformerNet()
        res = 224
else:
    raise NotImplementedError

model.eval()
if args.profile:
    # model = torch.compile(model.cuda())
    model.cuda()
    input_tensor = torch.rand(args.bs, 3, res, res).cuda()
    rounds = 500

    with torch.no_grad():
        # warm up
        for i in tqdm(range(rounds)):
            model(input_tensor)
            torch.cuda.synchronize()
        # running
        tot = 0
        for i in tqdm(range(rounds)):
            torch.cuda.synchronize()
            start = time()
            result = model(input_tensor)
            torch.cuda.synchronize()
            tot += time() - start
        print(tot / rounds * 1000, "ms")
else:
    model_name = f"{args.model}_bs{args.bs}_res{res}.onnx"
    buffer = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(model, torch.rand(args.bs, 3, res, res), buffer, opset_version=11)
        buffer.seek(0, 0)

        onnx_model = onnx.load_model(buffer)
        onnx_model, success = simplify(onnx_model)
        assert success
        new_buffer = io.BytesIO()
        onnx.save(onnx_model, new_buffer)
        buffer = new_buffer
        buffer.seek(0, 0)

    if buffer.getbuffer().nbytes > 0:
        with open(model_name, "wb") as f:
            f.write(buffer.read())
