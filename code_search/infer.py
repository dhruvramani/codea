import onnx
import onnxruntime

def infer(nl_text, code, onnx_model, tokenizer):
    assert type(nl_text) is list and type(code) is list
    f_inp = ['</s>'.join((nl, c)) for c in code for nl in nl_text]
    encoded_input = tokenizer.batch_encode_plus(f_inp, padding=True, return_tensors='pt')
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in encoded_input.items()}

    sft_logits = run_model(onnx_model, inputs_onnx['input_ids'], inputs_onnx['attention_mask'])
    return sft_logits.tolist()

def run_model(model, input_ids, attention_mask):
    ort_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask} 
    # OR ort_inputs = {model.get_inputs()[0].name: input_ids ...}

    ort_outs = model.run(None, ort_inputs)
    return ort_outs[0]

def get_model(path):
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)

    onnx_model = onnxruntime.InferenceSession(path)

    return onnx_model