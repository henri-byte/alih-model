BASE = 'Helsinki-NLP/opus-mt-ja-ms'
LANG_TOKEN = '>>ind<<'
CTX_SEP, CUR_SEP = ' ◇ ', ' ◆ '

def make_input(ja, ctx=()):
    """Teks masukan model: token bahasa, lalu maks. 2 kalimat sebelumnya sebagai konteks, lalu kalimat yang diterjemahkan."""
    ctx = [c for c in (ctx or []) if c][-2:]
    return LANG_TOKEN + ' ' + (CTX_SEP.join(ctx) + CUR_SEP if ctx else '') + ja

MANGA = ['おはよう！今日も遅刻だよ〜！', 'その日、すべてが変わった。', '待って、先輩！', 'お前は誰だ？', 'そんなこと、できるわけないだろ！',
         'ありがとう、助かったよ。', 'もう逃げられないぞ。', '俺が守ってやる！', 'なんでここにいるの？', '絶対に許さない…！', '腹減ったなぁ',
         'これが俺の本当の力だ！', 'べ、別にあんたのためじゃないんだからね！', 'やれやれ、面倒なことになったな。', 'お兄ちゃん、起きて！',
         'ここは俺に任せて先に行け！', 'まさか…あいつが犯人だったのか？', 'うるさい！黙ってろ！', '大丈夫？怪我はない？', 'またね！']
# pasangan dengan konteks: kalimat yang sama bisa berubah artinya
MANGA_CTX = [(['先輩、これ食べてもいいですか？'], 'いいよ。'), (['犯人はお前だ！'], '違う！俺じゃない！'),
             (['彼女は泣いていた。'], 'どうして…？'), (['俺の財布どこ？'], 'テーブルの上にあるよ。')]
