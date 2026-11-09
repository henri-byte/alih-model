"""Ekspor model terlatih ke ONNX (terkuantisasi int8) dalam format yang sama dengan Xenova/opus-mt-*,
buat tokenizer.json, lalu periksa: tokenisasi sama persis, dan hasil ONNX sama dengan PyTorch."""
import base64, gzip, json, os, shutil, sys, numpy as np, torch
from transformers import MarianMTModel, MarianTokenizer
from huggingface_hub import hf_hub_download
from common import BASE, make_input, MANGA, MANGA_CTX, LANG_TOKEN

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT + '/onnx', exist_ok=True)
tok = MarianTokenizer.from_pretrained(SRC)

# ---------- tokenizer.json (dipakai @huggingface/tokenizers di aplikasi) ----------
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb
tpl = json.load(open(hf_hub_download('Xenova/opus-mt-ja-en', 'tokenizer.json')))
print('templat', {k: (v if k != 'model' and k != 'normalizer' else '…') for k, v in tpl.items() if k != 'added_tokens'}, tpl.get('added_tokens'))
proto = pb.ModelProto(); proto.ParseFromString(open(os.path.join(SRC, 'source.spm'), 'rb').read())
score = {p.piece: p.score for p in proto.pieces}
vocab = tok.get_vocab(); inv = sorted(vocab.items(), key=lambda kv: kv[1])
assert [i for _, i in inv] == list(range(len(inv)))
low = min(score.values()) - 10
tj = json.loads(json.dumps(tpl))
tj['model']['vocab'] = [[p, float(score.get(p, low if p not in ('</s>', '<unk>', '<pad>') else 0.0))] for p, _ in inv]
tj['model']['unk_id'] = vocab['<unk>']
tj['normalizer'] = {'type': 'Precompiled', 'precompiled_charsmap': base64.b64encode(proto.normalizer_spec.precompiled_charsmap).decode()}
spec = {'</s>': vocab['</s>'], '<unk>': vocab['<unk>'], '<pad>': vocab['<pad>']}
tj['added_tokens'] = [{'id': i, 'content': t, 'single_word': False, 'lstrip': False, 'rstrip': False, 'normalized': False, 'special': True} for t, i in spec.items()]
tj['added_tokens'].append({'id': vocab[LANG_TOKEN], 'content': LANG_TOKEN, 'single_word': False, 'lstrip': False, 'rstrip': True, 'normalized': False, 'special': True})
eos = vocab['</s>']
tj['post_processor'] = {'type': 'TemplateProcessing',
  'single': [{'Sequence': {'id': 'A', 'type_id': 0}}, {'SpecialToken': {'id': '</s>', 'type_id': 0}}],
  'pair': [{'Sequence': {'id': 'A', 'type_id': 0}}, {'Sequence': {'id': 'B', 'type_id': 0}}, {'SpecialToken': {'id': '</s>', 'type_id': 0}}],
  'special_tokens': {'</s>': {'id': '</s>', 'ids': [eos], 'tokens': ['</s>']}}}
json.dump(tj, open(OUT + '/tokenizer.json', 'w'), ensure_ascii=False)
for f in ['tokenizer_config.json', 'config.json', 'generation_config.json', 'vocab.json', 'source.spm', 'target.spm']:
    if os.path.exists(os.path.join(SRC, f)): shutil.copy(os.path.join(SRC, f), OUT)
print('post_processor', tj['post_processor'])

from tokenizers import Tokenizer
fast = Tokenizer.from_file(OUT + '/tokenizer.json')
with gzip.open('data/test.jsonl.gz', 'rt', encoding='utf8') as f: test = [json.loads(l) for l in f]
samples = [make_input(x['ja'], x['ctx']) for x in test] + [make_input(j) for j in MANGA] + [make_input(j, c) for c, j in MANGA_CTX]
bad = 0
for s in samples:
    a = tok(s)['input_ids']; b = fast.encode(s).ids
    if a != b:
        bad += 1
        if bad <= 10: print('BEDA TOKEN', s, a, b)
print(f'tokenisasi sama: {len(samples) - bad}/{len(samples)}')
assert bad <= len(samples) * 0.01, 'tokenizer.json tidak cocok'

# ---------- ONNX ----------
from optimum.exporters.onnx import main_export
main_export(SRC, output='onnx-fp32', task='text2text-generation-with-past')
print(os.listdir('onnx-fp32'))
from onnxruntime.quantization import quantize_dynamic, QuantType
for n in ['encoder_model', 'decoder_model_merged']:
    quantize_dynamic(f'onnx-fp32/{n}.onnx', f'{OUT}/onnx/{n}_quantized.onnx', weight_type=QuantType.QUInt8, per_channel=False, reduce_range=False, extra_options={'EnableSubgraph': True})
    print(n, os.path.getsize(f'{OUT}/onnx/{n}_quantized.onnx') // 1e6, 'MB')

# ---------- cek: greedy ONNX (sama seperti di aplikasi) vs PyTorch ----------
import onnxruntime as ort, sacrebleu
cfg = json.load(open(OUT + '/config.json'))
E = ort.InferenceSession(f'{OUT}/onnx/encoder_model_quantized.onnx'); D = ort.InferenceSession(f'{OUT}/onnx/decoder_model_merged_quantized.onnx')
H, L = cfg['decoder_attention_heads'], cfg['decoder_layers']; dh = cfg['d_model'] // H
def onnx_tr(text, max_new=96):
    ids = np.array([fast.encode(text).ids], dtype=np.int64); am = np.ones_like(ids)
    hid = E.run(None, {'input_ids': ids, 'attention_mask': am})[0]
    past = {i.name: np.zeros((1, H, 0, dh), np.float32) for i in D.get_inputs() if i.name.startswith('past_key_values')}
    cur, out = cfg['decoder_start_token_id'], []
    names = [o.name for o in D.get_outputs()]
    for step in range(max_new):
        feeds = {'input_ids': np.array([[cur]], np.int64), 'encoder_attention_mask': am, 'encoder_hidden_states': hid, 'use_cache_branch': np.array([step > 0])}
        feeds.update(past)
        r = dict(zip(names, D.run(None, feeds)))
        lg = r['logits'][0, -1].copy(); lg[cfg['pad_token_id']] = -1e9
        seq = out
        if len(seq) >= 2:
            for i in range(len(seq) - 2):
                if seq[i] == seq[-2] and seq[i + 1] == seq[-1]: lg[seq[i + 2]] = -1e9
        nxt = int(lg.argmax())
        if nxt == cfg['eos_token_id']: break
        out.append(nxt); cur = nxt
        for i in range(L):
            past[f'past_key_values.{i}.decoder.key'] = r[f'present.{i}.decoder.key']; past[f'past_key_values.{i}.decoder.value'] = r[f'present.{i}.decoder.value']
            if step == 0:
                past[f'past_key_values.{i}.encoder.key'] = r[f'present.{i}.encoder.key']; past[f'past_key_values.{i}.encoder.value'] = r[f'present.{i}.encoder.value']
    return tok.decode(out, skip_special_tokens=True)
model = MarianMTModel.from_pretrained(SRC).eval()
def pt_tr(texts):
    with torch.no_grad():
        b = tok(texts, return_tensors='pt', padding=True); g = model.generate(**b, num_beams=1, max_new_tokens=96, no_repeat_ngram_size=3)
    return tok.batch_decode(g, skip_special_tokens=True)
sub = test[:400]
hyp_o = [onnx_tr(make_input(x['ja'], x['ctx'])) for x in sub]
hyp_p = sum([pt_tr([make_input(x['ja'], x['ctx']) for x in sub[k:k + 32]]) for k in range(0, len(sub), 32)], [])
ref = [x['id'] for x in sub]
print('UJI chrF onnx-int8 %.2f | pytorch %.2f | sama persis %d/%d' % (sacrebleu.corpus_chrf(hyp_o, [ref]).score, sacrebleu.corpus_chrf(hyp_p, [ref]).score, sum(a == b for a, b in zip(hyp_o, hyp_p)), len(sub)))
hyp_n = [onnx_tr(make_input(x['ja'])) for x in sub]
print('UJI tanpa konteks chrF onnx-int8 %.2f BLEU %.2f | dengan konteks BLEU %.2f' % (sacrebleu.corpus_chrf(hyp_n, [ref]).score, sacrebleu.corpus_bleu(hyp_n, [ref]).score, sacrebleu.corpus_bleu(hyp_o, [ref]).score))
for j in MANGA: print('MANGA', j, '→', onnx_tr(make_input(j)))
for c, j in MANGA_CTX: print('KONTEKS', c, j, '→', onnx_tr(make_input(j, c)))
