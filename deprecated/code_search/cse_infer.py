import onnx
import onnxruntime

BATCH_SIZE = 32

def infer(nl_text, code, onnx_model, tokenizer):
    assert type(nl_text) is list and type(code) is list
    all_inp = ['</s>'.join((nl, c)) for c in code for nl in nl_text]
    all_logits = []
    for i, f_inp in enumerate(chunks(all_inp, BATCH_SIZE)):
        encoded_input = tokenizer.batch_encode_plus(f_inp, padding=True, truncation=True, return_tensors='pt')
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in encoded_input.items()}

        sft_logits = run_model(onnx_model, inputs_onnx['input_ids'], inputs_onnx['attention_mask'])
        all_logits.extend(sft_logits.tolist())
    return all_logits

def run_model(model, input_ids, attention_mask):
    ort_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask} 
    ort_outs = model.run(None, ort_inputs)
    return ort_outs[0]

def get_model(path):
    onnx_model = onnxruntime.InferenceSession(path)

    return onnx_model

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
