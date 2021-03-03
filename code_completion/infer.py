import onnx
import onnxruntime

def infer(input_code, onnx_model, tokenizer):
    tokens = tokenizer(input_code) 
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in tokens.items()}

    outputs = [tokenizer.decode(gen_op, skip_special_tokens=True) for gen_op in gen_outputs]
    return outputs.tolist()

def run_model(model, input_ids):
    ort_inputs = {'input_ids': input_ids} 
    # OR ort_inputs = {model.get_inputs()[0].name: input_ids ...}

    ort_outs = model.run(None, ort_inputs)
    return ort_outs[0]

def get_model(path):
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)

    onnx_model = onnxruntime.InferenceSession(path)

    return onnx_model