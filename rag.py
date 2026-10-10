
import re
from collections import Counter
from math import log
from policy_loader import load_policy_documents

documents = load_policy_documents()

STOPWORDS = set([
    'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
    'is','are','was','were','be','been','being','have','has','had','do','does',
    'did','will','would','shall','should','may','might','must','can','could',
    'i','you','he','she','it','we','they','me','him','her','us','them','my',
    'your','his','its','our','their','this','that','these','those','am','not',
    'no','nor','so','if','then','else','when','where','why','how','all','any',
    'both','each','few','more','most','other','some','such','only','own','same',
    'than','too','very','just','about','above','after','again','against',
    'between','into','through','during','before','below','up','down','out',
    'off','over','under','further','once','here','there','now'
])

def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b[a-z0-9]+\b', text)
    return [w for w in words if w not in STOPWORDS and len(w) > 1]

doc_tokens = [tokenize(doc.page_content) for doc in documents]

N = len(documents)
df = Counter()
for tokens in doc_tokens:
    for word in set(tokens):
        df[word] += 1

idf = {word: log((N + 1) / (df[word] + 1)) + 1 for word in df}

def score_document(query, doc_idx):
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    doc_token_counts = Counter(doc_tokens[doc_idx])
    score = 0.0
    for word in query_tokens:
        if word in doc_token_counts:
            tf = doc_token_counts[word]
            score += tf * idf.get(word, 0.0)
    doc_len = len(doc_tokens[doc_idx]) or 1
    return score / doc_len

def retrieve_documents(query, top_k=5):
    scores = [(score_document(query, i), i) for i in range(N)]
    scores.sort(reverse=True, key=lambda x: x[0])
    top_indices = [i for score, i in scores[:top_k] if score > 0]
    return [documents[i] for i in top_indices]
