"""Siapkan data latih Jepang -> Indonesia dari korpus terbuka OPUS.
Tiap contoh: (konteks = maks. 2 kalimat Jepang sebelumnya, kalimat Jepang, terjemahan Indonesia)."""
import gzip, io, os, random, re, sys, unicodedata, urllib.request, zipfile, json

CORPORA = [  # nama, url zip moses, berurutan (punya konteks), bobot
    ('OpenSubtitles', 'https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/id-ja.txt.zip', True),
    ('TED2020', 'https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/id-ja.txt.zip', True),
    ('Tatoeba', 'https://object.pouta.csc.fi/OPUS-Tatoeba/v2026-07-08/moses/id-ja.txt.zip', False),
]
OUT = sys.argv[1] if len(sys.argv) > 1 else 'data'
os.makedirs(OUT, exist_ok=True)
JA_RE = re.compile(r'[぀-ヿ一-鿿]')
TAG = re.compile(r'<[^>]{1,20}>|\{[^}]{0,40}\}')

def clean(s):
    s = unicodedata.normalize('NFKC', s)
    s = TAG.sub('', s).replace('♪', '').replace('#', '')
    s = re.sub(r'^\s*[-–—/]\s*', '', s)           # tanda dialog di awal
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def ok(ja, idn):
    if not ja or not idn or len(ja) > 160 or len(idn) > 320: return False
    if not JA_RE.search(ja): return False
    if JA_RE.search(idn): return False
    letters = sum(c.isalpha() for c in idn)
    if letters < 0.5 * len(idn.replace(' ', '')): return False
    r = len(idn) / max(1, len(ja))
    if r < 0.8 or r > 7: return False
    if ' - ' in idn or '/' in idn or idn.startswith('-'): return False          # beberapa baris dialog tergabung
    if ('?' in ja) != ('?' in idn): return False                               # tanda tanya tidak cocok = kemungkinan salah jodoh
    if re.fullmatch(r'[\W\d_]*', re.sub(r'[A-Za-z]', '', ja)): return False  # sisi Jepang hanya nama/angka Latin
    if re.search(r'(https?:|www\.|@)', ja + idn): return False
    return True

def fetch(url):
    print('unduh', url, flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'alih-model'})
    return zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req, timeout=600).read()))

rows = {}
for name, url, seq in CORPORA:
    z = fetch(url)
    ja_f = [n for n in z.namelist() if n.endswith('.ja')][0]; id_f = [n for n in z.namelist() if n.endswith('.id')][0]
    ja = z.read(ja_f).decode('utf8', 'replace').split('\n'); idn = z.read(id_f).decode('utf8', 'replace').split('\n')
    out, prev, seen = [], [], set()
    for a, b in zip(ja, idn):
        a, b = clean(a), clean(b)
        if not ok(a, b):
            # pasangan rusak tidak dipakai, tapi kalimat Jepangnya tetap jadi konteks kalimat berikutnya
            prev = (prev + [a])[-2:] if a and JA_RE.search(a) else []
            continue
        key = (a, b)
        ctx = prev[-2:] if seq else []
        if key not in seen or ctx:
            out.append((ctx, a, b)); seen.add(key)
        prev = (prev + [a])[-2:]
    print(name, 'mentah', len(ja), 'bersih', len(out), flush=True)
    rows[name] = out

random.seed(7)
def split_block(lst, n, at):
    # ambil blok berurutan sebagai data uji supaya tidak bocor lewat konteks
    i = int(len(lst) * at); return lst[:i] + lst[i + n:], lst[i:i + n]
train, dev, test = [], [], []
for name, lst in rows.items():
    if name in ('OpenSubtitles', 'TED2020'):
        lst, d = split_block(lst, 1500 if name == 'OpenSubtitles' else 500, 0.37)
        lst, t = split_block(lst, 1500 if name == 'OpenSubtitles' else 500, 0.71)
        dev += [(name,) + x for x in d]; test += [(name,) + x for x in t]
    train += [(name,) + x for x in lst]
random.shuffle(train)

def dump(fn, data):
    with gzip.open(os.path.join(OUT, fn), 'wt', encoding='utf8') as f:
        for name, ctx, a, b in data: f.write(json.dumps({'c': name, 'ctx': ctx, 'ja': a, 'id': b}, ensure_ascii=False) + '\n')
dump('train.jsonl.gz', train); dump('dev.jsonl.gz', dev); dump('test.jsonl.gz', test)
print('train', len(train), 'dev', len(dev), 'test', len(test))
for x in random.sample(train, 25): print('CONTOH', x)
