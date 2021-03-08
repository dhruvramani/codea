import onnx
import onnxruntime

def infer(code, model, tokenizer):
    enc_input = tokenizer(code, return_tensors='pt')
    # print(enc_input['input_ids'], enc_input['attention_mask'])
    preds = run_model(model, enc_input['input_ids'], enc_input['attention_mask'])
    p = []
    for pred in preds:
        t = pred[0].cpu().numpy()
        t = list(t)
        if 0 in t:
            t = t[: t.index(0)]
        text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
        p.append(text)

    return p

def _model(tokenizer, pretrained_path):
    from models.p_codebert_model import get_model
    return get_model(tokenizer, pretrained_path=pretrained_path)

def run_model(model, input_ids, attention_mask):
    preds = model(source_ids=input_ids, source_mask=attention_mask)
    return preds

# def run_model(model, input_ids, attention_mask):
    # ort_outs = model(source_ids=input_ids, source_mask=attention_mask)
    # return ort_outs

# def get_model(encoder_path, decoder_path, dense_path, lm_path, tokenizer):
    # onnx_model = onnx.load(path)
    # onnx.checker.check_model(onnx_model)
    # from models.p_codebert_model import Seq2Seq
    # from transformers import RobertaConfig

    # encoder = onnxruntime.InferenceSession(encoder_path)
    # decoder = onnxruntime.InferenceSession(decoder_path)
    # dense = onnxruntime.InferenceSession(dense_path)
    # lm_head = onnxruntime.InferenceSession(lm_path)

    # model_config = RobertaConfig.from_pretrained('microsoft/codebert-base')
    # model_config.output_hidden_states = True

    # onnx_model = Seq2Seq(encoder, decoder, model_config, onnx=True, dense=dense, lm_head=lm_head, beam_size=10, \
    #         max_length=128, sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    # return onnx_model

# def convert_p_codebert_onnx(ckpt_dir):
    # import torch
    # from config import get_config
    # from train import select_model

    # config = get_config(checkpoint='pretrained.ckpt')
    # pl_model = select_model(config)

    # tokens = pl_model.tokenizer("This is a sample output", return_tensors='pt')
    # seq_len = tokens['input_ids'].shape[-1]


    # encoder_path = ckpt_dir + 'p_codebert-summ-encoder.onnx'
    # encoder_inputs = (tokens['input_ids'], tokens['attention_mask'])
    # encoder_outputs = pl_model.model.encoder(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
    # encoder_input_names = ['input_ids', 'attention_mask']
    # encoder_output_names = ['output_0', 'output_1', 'output_2']
    # encoder_dynamic_axes = {k : {0: 'batch_size', 1: 'seq_len'} for k in encoder_input_names + encoder_output_names}

    # torch.onnx.export(model=pl_model.model.encoder, args=encoder_inputs, f=encoder_path, input_names=encoder_input_names,
    #               output_names=encoder_output_names, example_outputs=encoder_outputs, dynamic_axes=encoder_dynamic_axes, do_constant_folding=True, opset_version=11, use_external_data_format=False)

    # print("Encoder exported @ ", encoder_path)


    # decoder_path = ckpt_dir + 'p_codebert-summ-decoder.onnx'
    # decoder_inputs = (torch.rand(1, 10, 768), torch.rand(seq_len, 10, 768), torch.rand(1, 1), torch.rand(10, seq_len).bool())
    # decoder_outputs = pl_model.model.decoder(decoder_inputs[0], decoder_inputs[1], decoder_inputs[2], decoder_inputs[3])
    # decoder_input_names = ['tgt', 'memory', 'tgt_mask', 'memory_key_padding_mask']
    # decoder_output_names = ['output_0']
    # decoder_dynamic_axes = {'tgt': {0: 'batch_size'}, 'output_0': {0: 'batch_size'}, 'memory': {0: 'seq_len', 1: 'batch_size'}, 'memory_key_padding_mask' : {1: 'seq_len'}, 'tgt_mask': {0: 'sth', 1: 'sth1'}}

    # torch.onnx.export(model=pl_model.model.decoder, args=decoder_inputs, f=decoder_path, input_names=decoder_input_names,
    #               output_names=decoder_output_names, example_outputs=decoder_outputs, dynamic_axes=decoder_dynamic_axes, do_constant_folding=True, opset_version=11, use_external_data_format=False)    

    # print("Decoder exported @ ", decoder_path)



    # dense_path = ckpt_dir + 'p_codebert-summ-dense.onnx'
    # dense_inputs = (torch.rand(1, 10, 768))
    # dense_outputs = pl_model.model.dense(dense_inputs[0])
    # dense_input_names = ['input']
    # dense_output_names = ['output_0']
    # dense_dynamic_axes = {'input': {0: 'batch_size'}, 'output_0': {0: 'batch_size'}}

    # torch.onnx.export(model=pl_model.model.dense, args=dense_inputs, f=dense_path, input_names=dense_input_names,
    #               output_names=dense_output_names, example_outputs=dense_outputs, dynamic_axes=dense_dynamic_axes, do_constant_folding=True, opset_version=11, use_external_data_format=False)    

    # print("Dense exported @ ", dense_path)


    # lm_head_path = ckpt_dir + 'p_codebert-summ-lm_head.onnx'
    # lm_head_inputs = (torch.rand(10, 768))
    # lm_head_outputs = pl_model.model.lm_head(lm_head_inputs[0])
    # lm_head_input_names = ['input']
    # lm_head_output_names = ['output_0']
    # lm_head_dynamic_axes = None

    # torch.onnx.export(model=pl_model.model.lm_head, args=lm_head_inputs, f=lm_head_path, input_names=lm_head_input_names,
    #               output_names=lm_head_output_names, example_outputs=lm_head_outputs, dynamic_axes=lm_head_dynamic_axes, do_constant_folding=True, opset_version=11, use_external_data_format=False)

    # print("LM Head exported @ ", lm_head_path)
    # return [encoder_path, decoder_path, dense_path, lm_head_path]
    # return [decoder_path]

# def optimize_quantize_models(path_list):
    # import os, sys, argparse

    # sys.path.append("/content/drive/My Drive/Startup/code/")
    # from prod_onnx import optimize_onnx, quantize_onnx

    # for path in path_list:
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--onnx_save_path', type=str, default=path)
    #     config, _ = parser.parse_known_args()

    #     optimize_onnx(config)
    #     quantize_onnx(config)

# if __name__ == '__main__':
#     ckpt_dir = "/content/drive/My Drive/Startup/save/onnx_models/code_summ/"
#     path_list = convert_p_codebert_onnx(ckpt_dir)
#     optimize_quantize_models(path_list)


# tgt_embeddings - permuted  torch.Size([1, 10, 768])
# context, attn_mask, context_mask.bool  torch.Size([24, 10, 768]) torch.Size([1, 1]) torch.Size([10, 24])
# decoder_out  torch.Size([1, 10, 768])
# dense_out  torch.Size([1, 10, 768])
# hidden_states  torch.Size([10, 768])
# lm_out  torch.Size([10, 50265])


# source_ids  torch.Size([1, 24])
# outputs  torch.Size([1, 24, 768])
# input_ids  torch.Size([10, 1])
# tgt_embeddings - from enc op torch.Size([10, 1, 768])
# tgt_embeddings - permuted  torch.Size([1, 10, 768])
# context, attn_mask, context_mask.bool  torch.Size([24, 10, 768]) torch.Size([1, 1]) torch.Size([10, 24])
# decoder_out  torch.Size([1, 10, 768])
# dense_out  torch.Size([1, 10, 768])
# hidden_states  torch.Size([10, 768])
# lm_out  torch.Size([10, 50265])

# source_ids  torch.Size([1, 19])
# outputs  torch.Size([1, 19, 768])
# input_ids  torch.Size([10, 1])
# tgt_embeddings - from enc op torch.Size([10, 1, 768])
# tgt_embeddings - permuted  torch.Size([1, 10, 768])
# context, attn_mask, context_mask.bool  torch.Size([19, 10, 768]) torch.Size([1, 1]) torch.Size([10, 19])
# decoder_out  torch.Size([1, 10, 768])
# dense_out  torch.Size([1, 10, 768])
# hidden_states  torch.Size([10, 768])
# lm_out  torch.Size([10, 50265])