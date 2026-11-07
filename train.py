"""Latih ulang (fine-tune) opus-mt-ja-ms menjadi penerjemah Jepang -> Indonesia yang paham konteks.
Dijalankan berantai: tiap job berlatih sampai batas waktu, menyimpan checkpoint, lalu job berikutnya melanjutkan."""
import gzip, json, math, os, random, sys, time, torch
from transformers import MarianMTModel, MarianTokenizer
from common import BASE, make_input, MANGA, MANGA_CTX

DATA, CKPT = os.environ.get('DATA', 'data'), os.environ.get('CKPT', 'ckpt')
BUDGET = float(os.environ.get('BUDGET_MIN', '300')) * 60
TOTAL = int(os.environ.get('TOTAL_STEPS', '60000'))
BS, LR, WARM = int(os.environ.get('BS', '32')), float(os.environ.get('LR', '5e-5')), 1000
P_NOCTX = 0.3     # sebagian contoh tanpa konteks supaya model juga bisa dipakai kalimat tunggal
torch.set_num_threads(os.cpu_count()); torch.manual_seed(0)
t0 = time.time()

tok = MarianTokenizer.from_pretrained(BASE)
state = None
if os.path.exists(os.path.join(CKPT, 'state.pt')):
    model = MarianMTModel.from_pretrained(os.path.join(CKPT, 'model'))
    state = torch.load(os.path.join(CKPT, 'state.pt'), weights_only=False)
    print('lanjut dari langkah', state['step'], flush=True)
else:
    model = MarianMTModel.from_pretrained(BASE)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / WARM) * max(0.05, 1 - s / TOTAL))
step, best = 0, -1
if state:
    opt.load_state_dict(state['opt']); sched.load_state_dict(state['sched']); step, best = state['step'], state.get('best', -1)

def load(fn):
    with gzip.open(os.path.join(DATA, fn), 'rt', encoding='utf8') as f: return [json.loads(l) for l in f]
train = load('train.jsonl.gz'); dev = load('dev.jsonl.gz')
print('data', len(train), 'dev', len(dev), flush=True)
rng = random.Random(1234)

def batches():
    # urutan acak tetap per putaran (bisa dilanjutkan), dikelompokkan menurut panjang supaya padding sedikit
    ep, off = divmod(step * BS, len(train))
    while True:
        idx = list(range(len(train))); random.Random(ep).shuffle(idx); idx = idx[off:]; off = 0
        for k in range(0, len(idx), BS * 50):
            chunk = sorted(idx[k:k + BS * 50], key=lambda i: len(train[i]['ja']) + sum(len(c) for c in train[i]['ctx']))
            bs = [chunk[j:j + BS] for j in range(0, len(chunk), BS)]; rng.shuffle(bs)
            yield from bs
        ep += 1

def encode(items):
    src = [make_input(x['ja'], [] if rng.random() < P_NOCTX else x['ctx']) for x in items]
    enc = tok(src, text_target=[x['id'] for x in items], max_length=160, truncation=True, padding=True, return_tensors='pt')
    enc['labels'][enc['labels'] == tok.pad_token_id] = -100
    return enc

@torch.no_grad()
def translate(srcs, bs=32):
    model.eval(); out = []
    for k in range(0, len(srcs), bs):
        b = tok(srcs[k:k + bs], return_tensors='pt', padding=True, truncation=True, max_length=160)
        g = model.generate(**b, num_beams=1, max_new_tokens=96, no_repeat_ngram_size=3)
        out += tok.batch_decode(g, skip_special_tokens=True)
    model.train(); return out

def evaluate(n=600):
    import sacrebleu
    sub = dev[:n]
    hyp = translate([make_input(x['ja'], x['ctx']) for x in sub]); ref = [x['id'] for x in sub]
    chrf = sacrebleu.corpus_chrf(hyp, [ref]).score; bleu = sacrebleu.corpus_bleu(hyp, [ref]).score
    hyp0 = translate([make_input(x['ja']) for x in sub[:300]]); chrf0 = sacrebleu.corpus_chrf(hyp0, [ref[:300]]).score
    print(f'EVAL langkah {step}: chrF {chrf:.2f} BLEU {bleu:.2f} (tanpa konteks chrF {chrf0:.2f})', flush=True)
    for j, h in zip(MANGA, translate([make_input(j) for j in MANGA])): print('  MANGA', j, '→', h)
    for (c, j), h in zip(MANGA_CTX, translate([make_input(j, c) for c, j in MANGA_CTX])): print('  KONTEKS', c, j, '→', h)
    return chrf

def save(tag='model'):
    os.makedirs(CKPT, exist_ok=True)
    model.save_pretrained(os.path.join(CKPT, tag)); tok.save_pretrained(os.path.join(CKPT, tag))

if step == 0: evaluate(300)   # nilai awal model dasar sebagai pembanding
it = batches(); loss_avg, last = 0.0, time.time()
while step < TOTAL and time.time() - t0 < BUDGET:
    enc = encode([train[i] for i in next(it)])
    loss = model(**enc).loss
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
    loss_avg = 0.98 * loss_avg + 0.02 * loss.item() if step > 1 else loss.item()
    if step % 200 == 0:
        print(f'langkah {step}/{TOTAL} loss {loss_avg:.3f} lr {sched.get_last_lr()[0]:.2e} {200 / (time.time() - last):.2f} langkah/dtk', flush=True); last = time.time()
chrf = evaluate()
if chrf > best: best = chrf; save('best'); print('model terbaik baru', flush=True)
save('model')
torch.save({'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step, 'best': best}, os.path.join(CKPT, 'state.pt'))
open(os.path.join(CKPT, 'DONE' if step >= TOTAL else 'MORE'), 'w').write(str(step))
print('selesai job, langkah', step, 'terbaik', best)
