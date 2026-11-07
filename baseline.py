"""Nilai pembanding: jalur lama aplikasi (ja→en→id, Opus-MT) dan model dasar ja-ms, pada data uji yang sama."""
import gzip, json, sys, torch, sacrebleu
from transformers import MarianMTModel, MarianTokenizer
from common import make_input
with gzip.open('data/test.jsonl.gz', 'rt', encoding='utf8') as f: test = [json.loads(l) for l in f]
sub = test[:400]; ref = [x['id'] for x in sub]
def tr(name, texts):
    tok = MarianTokenizer.from_pretrained(name); m = MarianMTModel.from_pretrained(name).eval(); out = []
    with torch.no_grad():
        for k in range(0, len(texts), 32):
            g = m.generate(**tok(texts[k:k + 32], return_tensors='pt', padding=True, truncation=True), num_beams=1, max_new_tokens=96, no_repeat_ngram_size=3)
            out += tok.batch_decode(g, skip_special_tokens=True)
    return out
piv = tr('Helsinki-NLP/opus-mt-en-id', tr('Helsinki-NLP/opus-mt-ja-en', [x['ja'] for x in sub]))
base = tr('Helsinki-NLP/opus-mt-ja-ms', [make_input(x['ja']) for x in sub])
for n, h in [('jalur lama ja→en→id', piv), ('dasar ja-ms', base)]:
    print('PEMBANDING', n, 'chrF %.2f BLEU %.2f' % (sacrebleu.corpus_chrf(h, [ref]).score, sacrebleu.corpus_bleu(h, [ref]).score))
