import torch
from pytorch_model import Classifier, BasicBlock
import torch.onnx
import os
print(f"Current working directory at start of main: {os.getcwd()}")
ONNX_MODEL_PATH = "mtailor_model.onnx"

mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
mtailor.load_state_dict(torch.load("./resnet18-f37072fd.pth"))
mtailor.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    mtailor,
    dummy_input,
    "mtailor_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("Exported to mtailor_model.onnx")

