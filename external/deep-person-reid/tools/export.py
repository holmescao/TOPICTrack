import argparse
import os
import sys
import numpy as np
from pathlib import Path
import torch
import pandas as pd
import subprocess

from torchreid.utils.feature_extractor import FeatureExtractor
from torchreid.models import build_model

__model_types = [
    "resnet50",
    "mlfn",
    "hacnn",
    "mobilenetv2_x1_0",
    "mobilenetv2_x1_4",
    "osnet_x1_0",
    "osnet_x0_75",
    "osnet_x0_5",
    "osnet_x0_25",
    "osnet_ibn_x1_0",
    "osnet_ain_x1_0",
]


def file_size(path):

    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / 1e6
    else:
        return 0.0


def get_model_name(model):
    model = str(model).rsplit("/", 1)[-1].split(".")[0]
    for x in __model_types:
        if x in model:
            return x
    return None


def export_formats():
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def export_onnx(model, im, file, opset, train=False, dynamic=True, simplify=False):
    try:
        import onnx

        f = file.with_suffix(".onnx")
        print(f"\nStarting export with onnx {onnx.__version__}...")

        torch.onnx.export(
            model.cpu() if dynamic else model, 
            im.cpu() if dynamic else im,
            f,
            verbose=False,
            opset_version=opset,
            training=torch.onnx.TrainingMode.TRAINING
            if train
            else torch.onnx.TrainingMode.EVAL,
            do_constant_folding=not train,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={
                "images": {
                    0: "batch",
                }, 
                "output": {
                    0: "batch",
                }, 
            }
            if dynamic
            else None,
        )

        model_onnx = onnx.load(f) 
        onnx.checker.check_model(model_onnx) 
        onnx.save(model_onnx, f)

        if simplify:
            try:
                cuda = torch.cuda.is_available()
                import onnxsim

                print(f"simplifying with onnx-simplifier {onnxsim.__version__}...")
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={"t0": list(im.shape)} if dynamic else None,
                )
                assert check, "assert check failed"
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f"simplifier failure: {e}")
        print(f"export success, saved as {f} ({file_size(f):.1f} MB)")
        print(
            f"run --dynamic ONNX model inference with: 'python detect.py --weights {f}'"
        )
    except Exception as e:
        print(f"export failure: {e}")
    return f


def export_openvino(file, dynamic, half):
    f = str(file).replace(".onnx", f"_openvino_model{os.sep}")

    try:
        import openvino.inference_engine as ie

        print(f"\nStarting export with openvino {ie.__version__}...")
        f = str(file).replace(".onnx", f"_openvino_model{os.sep}")
        dyn_shape = [-1, 3, 256, 128] if dynamic else None
        cmd = f"mo \
            --input_model {file} \
            --output_dir {f} \
            --data_type {'FP16' if half else 'FP32'}"

        if dyn_shape is not None:
            cmd + f"--input_shape {dyn_shape}"

        subprocess.check_output(cmd.split())

        print(f"Export success, saved as {f} ({file_size(f):.1f} MB)")
        return f
    except Exception as e:
        print(f"\nExport failure: {e}")
    return f


def export_tflite(file, half):
    try:
        import openvino.inference_engine as ie

        print(f"\nStarting export with openvino {ie.__version__}...")
        output = Path(
            str(file).replace(f"_openvino_model{os.sep}", f"_tflite_model{os.sep}")
        )
        modelxml = list(Path(file).glob("*.xml"))[0]
        cmd = f"openvino2tensorflow \
            --model_path {modelxml} \
            --model_output_path {output} \
            --output_pb \
            --output_saved_model \
            --output_no_quant_float32_tflite \
            --output_dynamic_range_quant_tflite"
        subprocess.check_output(cmd.split())

        print(f"Export success, results saved in {output} ({file_size(f):.1f} MB)")
        return f
    except Exception as e:
        print(f"\nExport failure: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CPHD train")
    parser.add_argument(
        "-d",
        "--dynamic",
        action="store_true",
        help="dynamic model input",
    )
    parser.add_argument(
        "-p",
        "--weights",
        type=Path,
        default="./mobilenetv2_x1_0_msmt17.pt",
        help="Path to weights",
    )
    parser.add_argument(
        "-hp",
        "--half_precision",
        action="store_true",
        help="transform model to half precision",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[256, 128],
        help="image (h, w)",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=["onnx", "openvino", "tflite"],
        help="onnx, openvino, tflite",
    )
    args = parser.parse_args()

    extractor = FeatureExtractor(

        model_name=get_model_name(args.weights),
        model_path=args.weights,
        device=str("cpu"),
    )

    include = [x.lower() for x in args.include] 
    fmts = tuple(export_formats()["Argument"][1:])
    flags = [x in include for x in fmts]
    assert sum(flags) == len(
        include
    ), f"ERROR: Invalid --include {include}, valid --include arguments are {fmts}"
    onnx, openvino, tflite = flags

    im = torch.zeros(1, 3, args.imgsz[0], args.imgsz[1]).to(
        "cpu"
    ) 
    if onnx:
        f = export_onnx(
            extractor.model.eval(),
            im,
            args.weights,
            12,
            train=False,
            dynamic=args.dynamic,
            simplify=True,
        )
    if openvino:
        f = export_openvino(f, dynamic=args.dynamic, half=False)
    if tflite:
        export_tflite(f, False)
