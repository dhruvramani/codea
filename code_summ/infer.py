import onnx
import onnxruntime

def infer(code, onnx_model, tokenizer):
    enc_input = tokenizer(code, return_tensors='pt')
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in enc_input.items()}

    preds = run_model(onnx_model, inputs_onnx['input_ids'], inputs_onnx['attention_mask'])

    p = []
    for pred in preds:
        t = pred[0].cpu().numpy()
        t = list(t)
        if 0 in t:
            t = t[: t.index(0)]
        text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
        p.append(text)

    return p

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

# Found input input_ids with shape: {0: 'batch', 1: 'sequence'}
# Found input attention_mask with shape: {0: 'batch', 1: 'sequence'}
# Found output output_0 with shape: {0: 'batch'}
# preds is not present in the generated input list.
# Generated inputs order: ['input_ids', 'attention_mask']

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
# cpu_model = create_model_for_provider("onnx/bert-base-cased.onnx", "CPUExecutionProvider")

# model_inputs = tokenizer("My name is Bert", return_tensors="pt")
# inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

# # Run the model (None = get all the outputs)
# sequence, pooled = cpu_model.run(None, inputs_onnx)