import onnx
import onnxruntime

def infer(code, onnx_model, tokenizer):
    enc_input = tokenizer(code, return_tensors='pt')
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
    ort_outs = model(source_ids=input_ids, source_mask=attention_mask)
    return ort_outs

def get_model(encoder_path, decoder_path, dense_path, lm_path, tokenizer):
    # onnx_model = onnx.load(path)
    # onnx.checker.check_model(onnx_model)
    from models import Seq2Seq
    from transformers import RobertaConfig

    encoder = onnxruntime.InferenceSession(encoder_path)
    decoder = onnxruntime.InferenceSession(decoder_path)
    dense = onnxruntime.InferenceSession(dense_path)
    lm_head = onnxruntime.InferenceSession(lm_path)

    model_config = RobertaConfig.from_pretrained('microsoft/codebert-base')
    model_config.output_hidden_states = True

    onnx_model = Seq2Seq(encoder, decoder, model_config, onnx=True, dense=dense, lm_head=lm_head, beam_size=10, \
            max_length=128, sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    return onnx_model

def convert_p_codebert_onnx(ckpt_dir):
    


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