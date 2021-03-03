import os
import sys
import argparse

import torch.onnx

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
SAVE_DIR  = "/content/drive/My Drive/Startup/save/"

from code_summ.config import create_dir, str2bool

def get_config(def_module='code_search', def_model='p_codebert', checkpoint=None, save_dir=None):
    save_dir = save_dir if save_dir is not None else SAVE_DIR
    parser = argparse.ArgumentParser("Onnx - Module Independent Config.",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--quantize', type=str2bool, default=True)
    parser.add_argument('--module', type=str.lower, default=def_module, choices=['code_search', 'code_summ', 'code_completion'])
    parser.add_argument('--model', type=str.lower, default=def_model)

    parser.add_argument('--ckpt_file', type=str, default=checkpoint)
    parser.add_argument('--onnx_save_path', type=str, default=os.path.join(save_dir, 'onnx_models/'))

    config, _ = parser.parse_known_args()
    
    #config.ckpt_file = "epoch=7-step=206087.ckpt" # NOTE - REMOVE & CHANGE NAME
    ckpt_name = config.ckpt_file.split('.')[0] if config.ckpt_file else None
    
    config.onnx_save_path = os.path.join(config.onnx_save_path, f'{config.module}/')
    create_dir(config.onnx_save_path, recreate=False)
    config.onnx_save_path = os.path.join(config.onnx_save_path, '{}-{}-{}.onnx'.format(config.module, config.model, ckpt_name))

    return config

def onnx_main():
    config = get_config()
    convert_to_onnx(config)
    
    if config.quantize:
        optimize_onnx(config)
        quantize_onnx(config)

def convert_to_onnx(config):
    model, tokenizer = get_model_tokenizer(config)
    input_names, input_sample, output_names, dynamic_axes = get_input_config(model, tokenizer)

    # print(input_names, input_sample, dynamic_axes)
    # # print(model)
    # _ = input(" foo ")
    torch.onnx.export(model, input_sample, f=config.onnx_save_path, export_params=True,\
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,\
            do_constant_folding=True, enable_onnx_checker=True, opset_version=11, use_external_data_format=False)
    
    print("Converted to Onnx @ ", config.onnx_save_path)

def quantize_onnx(config):
    import onnx
    from onnxruntime.quantization import QuantizationMode, quantize

    quant_model_path = config.onnx_save_path.split('.')[0] + '-quant.onnx'
    onnx_model = onnx.load(config.onnx_save_path)

    quantized_model = quantize(model=onnx_model, quantization_mode=QuantizationMode.IntegerOps,\
                                force_fusions=True, symmetric_weight=True)
    onnx.save_model(quantized_model, quant_model_path)
    config.onnx_save_path = quant_model_path

    print("Optimized @ ", config.onnx_save_path)

def optimize_onnx(config):
    from onnxruntime import InferenceSession, SessionOptions

    opt_model_path = config.onnx_save_path.split('.')[0] + '-optim.onnx'
    sess_option = SessionOptions()
    sess_option.optimized_model_filepath = opt_model_path

    _ = InferenceSession(config.onnx_save_path, sess_option)
    config.onnx_save_path = opt_model_path

    print("Optimized @ ", config.onnx_save_path)

def get_model_tokenizer(onnx_config):
    from dataset_scripts import get_tokenizer

    if onnx_config.module == 'code_search':
        sys.path.append(os.path.expanduser('./code_search'))
        from code_search.config import get_config
        from code_search.train import select_model

    elif onnx_config.module == 'code_summ':
        sys.path.append(os.path.expanduser('./code_summ'))
        from code_summ.config import get_config
        from code_summ.train import select_model
    
    elif onnx_config.module == 'code_completion':
        sys.path.append(os.path.expanduser('./code_completion'))
        from code_completion.config import get_config
        from code_completion.train import select_model
    else:
        raise NotImplementedError

    model_config = get_config(def_model=onnx_config.model, checkpoint=onnx_config.ckpt_file)
    tokenizer = get_tokenizer(model_config)
    model = select_model(model_config, tokenizer)
    model.eval()

    print("Model loaded from ", model_config.resume_ckpt)
    return model, tokenizer

def get_input_config(model, tokenizer):
    from transformers.file_utils import ModelOutput

    tokens = tokenizer("This is a sample output", return_tensors='pt')
    seq_len = tokens['input_ids'].shape[-1]
    outputs = model(tokens['input_ids'], tokens['attention_mask'])

    if isinstance(outputs, ModelOutput):
        outputs = outputs.to_tuple()
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)

    input_vars = list(tokens.keys())
    input_dynamic_axes = {k: _build_shape_dict(k, v, True, seq_len) for k, v in tokens.items()}

    outputs_flat = []
    for output in outputs:
        if isinstance(output, (tuple, list)):
            outputs_flat.extend(output)
        else:
            outputs_flat.append(output)

    output_names = [f"output_{i}" for i in range(len(outputs_flat))]
    output_dynamic_axes = {k: _build_shape_dict(k, v, False, seq_len) for k, v in zip(output_names, outputs_flat)}

    # Create the aggregated axes representation
    dynamic_axes = dict(input_dynamic_axes, **output_dynamic_axes)

    ordered_input_names, model_args = _order_input_args(model, tokens, input_vars)
    
    return ordered_input_names, model_args, output_names, dynamic_axes


def _build_shape_dict(name, tensor, is_input, seq_len):
    # Source : https://github.com/huggingface/transformers/blob/0c2325198fd638e5d1f0c7dcbdd8bf7f14c0ff7d/src/transformers/convert_graph_to_onnx.py#L177
    if isinstance(tensor, (tuple, list)):
            return [_build_shape_dict(name, t, is_input, seq_len) for t in tensor]
    else:
        # Let's assume batch is the first axis with only 1 element (~~ might not be always true ...)
        axes = {[axis for axis, numel in enumerate(tensor.shape) if numel == 1][0]: "batch"}
        if is_input:
            if len(tensor.shape) == 2:
                axes[1] = "sequence"
            else:
                raise ValueError(f"Unable to infer tensor axes ({len(tensor.shape)})")
        else:
            seq_axes = [dim for dim, shape in enumerate(tensor.shape) if shape == seq_len]
            axes.update({dim: "sequence" for dim in seq_axes})

    print(f"Found {'input' if is_input else 'output'} {name} with shape: {axes}")
    return axes

def _order_input_args(model, tokens, input_names):
    # Source : https://github.com/huggingface/transformers/blob/0c2325198fd638e5d1f0c7dcbdd8bf7f14c0ff7d/src/transformers/convert_graph_to_onnx.py#L133
    model_args_name = model.forward.__code__.co_varnames
    model_args, ordered_input_names = [], []
    for arg_name in model_args_name[1:]:  # start at index 1 to skip "self" argument
        if arg_name in input_names:
            ordered_input_names.append(arg_name)
            model_args.append(tokens[arg_name])
        else:
            print(f"{arg_name} is not present in the generated input list.")
            break

    print("Generated inputs order: {}".format(ordered_input_names))
    return ordered_input_names, tuple(model_args)

if __name__ == '__main__':
    onnx_main()