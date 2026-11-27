"""
Retrieval evaluation for the KMi Stress-Sense assistant.
Replicates the deployed TF-IDF retriever (rag_pipeline.py) exactly, and compares it to
(a) the same retriever with query stopwords removed and (b) Okapi BM25.
Gold chunks are chunks that contain a pre-specified answer phrase.
"""
import glob, json, math, re, statistics, time, random
from langchain_text_splitters import RecursiveCharacterTextSplitter

K = 4

def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())

# ---- build chunks exactly as deployed (500 chars, 50 overlap) ----
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for f in sorted(glob.glob('documents/*.txt')):
    txt = open(f, encoding='utf-8').read()
    for c in splitter.split_text(txt):
        chunks.append({"source": f.split('/')[-1], "content": c, "tokens": tokenize(c)})
N = len(chunks)

# ---- deployed TF-IDF index ----
df = {}
for d in chunks:
    for t in set(d["tokens"]):
        df[t] = df.get(t, 0) + 1
idf = {t: math.log(N / (n + 1)) for t, n in df.items()}

def tfidf_scores(query_tokens):
    scores = []
    for d in chunks:
        L = len(d["tokens"])
        tf = {}
        for t in d["tokens"]:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in query_tokens:
            if t in tf:
                s += (tf[t] / L) * idf.get(t, 0)
        scores.append(s)
    return scores

# ---- BM25 ----
avgdl = sum(len(d["tokens"]) for d in chunks) / N
def bm25_scores(query_tokens, k1=1.5, b=0.75):
    scores = []
    for d in chunks:
        L = len(d["tokens"])
        tf = {}
        for t in d["tokens"]:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in query_tokens:
            if t in tf:
                n = df.get(t, 0)
                w = math.log(1 + (N - n + 0.5) / (n + 0.5))
                f = tf[t]
                s += w * f * (k1 + 1) / (f + k1 * (1 - b + b * L / avgdl))
        scores.append(s)
    return scores

STOP = set("""a an the is are was were be been being am do does did how what why when where which who whom whose
i me my we our you your it its he she they them their this that these those of in on at to for from by with
about as into than then so if or and but not no can could would should may might will shall just up out over
under again there here has have had having too very some any each both few more most other such only own same""".split())

def rank(scores):
    order = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [i for i, _ in order]

def run(system, q):
    qt = tokenize(q)
    if system == "tfidf":
        return rank(tfidf_scores(qt))
    if system == "tfidf_stop":
        return rank(tfidf_scores([t for t in qt if t not in STOP]))
    if system == "bm25":
        return rank(bm25_scores(qt))

# ---- question set: (group, question, gold answer phrase) ----
Q = [
 ("lexical","What does GSR measure?","measures electrical conductivity of the skin"),
 ("lexical","How is the personal baseline built?","spends the first week learning"),
 ("lexical","What is a confidence score?","confidence score between 0 and 1"),
 ("lexical","What are feature weights?","Feature weights tell you how much each sensor reading influenced"),
 ("lexical","What does low heart rate variability indicate?","Lower HRV (less variation)"),
 ("lexical","What is a neural network?","loosely inspired by the human brain"),
 ("lexical","What is retrieval augmented generation?","RAG is a technique where an LLM is given access"),
 ("lexical","What are the limitations of the wearable device?","research prototype and not a medical device"),
 ("lexical","Which microcontroller does the wearable use?","built on an ESP32 microcontroller"),
 ("lexical","When was the Transformer architecture introduced?","2017: The Transformer architecture is introduced"),
 ("lexical","How does a classifier learn from training data?","Training data: The model is shown many examples"),
 ("lexical","What does the accelerometer do in the device?","captures movement data to filter out motion artifacts"),
 ("paraphrase","Why do damp palms matter to the band?","which increases when a person sweats due to stress"),
 ("paraphrase","How long before the band knows what is normal for me?","spends the first week learning"),
 ("paraphrase","How sure is the gadget about its verdict?","confidence score between 0 and 1"),
 ("paraphrase","Which body signal had the biggest say in the result?","Feature weights tell you how much each sensor reading influenced"),
 ("paraphrase","Does the system learn the way a brain does?","loosely inspired by the human brain"),
 ("paraphrase","Can I rely on this for medical decisions?","research prototype and not a medical device"),
 ("paraphrase","Why might a workout make the reading wrong?","Physical exercise can cause elevated readings that are not stress-related"),
 ("paraphrase","How did chatbots like this come about historically?","2017: The Transformer architecture is introduced"),
 ("paraphrase","Why would two people get different alerts for the same pulse?","defined uniquely per person, not against a population average"),
 ("paraphrase","What stops the AI from just making things up?","a more accurate, grounded answer"),
 ("paraphrase","What happens inside the model between sensor input and prediction?","passes through the layers where it is transformed and weighted"),
 ("paraphrase","Is a high-percentage output a guarantee?","AI models can be wrong, even when confident"),
]

norm = lambda s: " ".join(s.split())
gold = []
for g, q, phrase in Q:
    ids = [i for i, c in enumerate(chunks) if norm(phrase) in norm(c["content"])]
    assert ids, f"gold phrase not found in any chunk: {phrase}"
    gold.append(set(ids))

systems = ["tfidf", "tfidf_stop", "bm25"]
res = {s: [] for s in systems}
for s in systems:
    for (g, q, p), gs in zip(Q, gold):
        order = run(s, q)
        first = next(r for r, i in enumerate(order, 1) if i in gs)
        res[s].append((g, q, first))

def summarise(s, group=None):
    rows = [r for r in res[s] if group is None or r[0] == group]
    n = len(rows)
    hit4 = sum(1 for r in rows if r[2] <= K) / n
    hit1 = sum(1 for r in rows if r[2] == 1) / n
    mrr = sum(1 / r[2] for r in rows) / n
    return n, hit1, hit4, mrr

print("N chunks:", N, " random Recall@4 expectation ~ %.3f" % (K / N))
print("\nSystem        group       n   Hit@1  Recall@4   MRR")
for s in systems:
    for grp in [None, "lexical", "paraphrase"]:
        n, h1, h4, m = summarise(s, grp)
        print(f"{s:12s}  {str(grp or 'all'):10s} {n:3d}  {h1:5.2f}   {h4:5.2f}    {m:5.2f}")

print("\nPer-question first-gold rank (deployed tfidf | tfidf_stop | bm25):")
for i, (g, q, p) in enumerate(Q):
    print(f"{i+1:2d} [{g[:3]}] {res['tfidf'][i][2]:2d} | {res['tfidf_stop'][i][2]:2d} | {res['bm25'][i][2]:2d}  {q}")

# position of gold chunk inside the top-K context (deployed system), hits only
pos = {1:0,2:0,3:0,4:0}
for g,q,r in res['tfidf']:
    if r <= K: pos[r]+=1
print("\nDeployed TF-IDF: rank of first gold chunk among hits:", pos)

# ---- footprint ----
t0=time.perf_counter()
chunks2 = []
for f in sorted(glob.glob('documents/*.txt')):
    txt = open(f, encoding='utf-8').read()
    for c in splitter.split_text(txt):
        chunks2.append({"content": c, "tokens": tokenize(c)})
dfx = {}
for d in chunks2:
    for t in set(d["tokens"]):
        dfx[t] = dfx.get(t,0)+1
idfx = {t: math.log(len(chunks2)/(n+1)) for t,n in dfx.items()}
json.dump({"corpus":[{"content":d["content"],"tokens":d["tokens"]} for d in chunks2],"idf":idfx}, open("simple_db.json","w"))
build_ms=(time.perf_counter()-t0)*1000
import os
print("\nIndex build+write: %.1f ms ; simple_db.json size: %.1f KB" % (build_ms, os.path.getsize("simple_db.json")/1024))

lat=[]
for _ in range(200):
    for (g,q,p) in Q:
        t=time.perf_counter(); run("tfidf", q); lat.append((time.perf_counter()-t)*1000)
lat.sort()
print("Query latency (deployed tfidf, ms): median %.3f  p95 %.3f  max %.3f  (n=%d)" % (statistics.median(lat), lat[int(0.95*len(lat))], lat[-1], len(lat)))
