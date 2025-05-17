# Required libraries:
# pip install datasets pandas matplotlib seaborn nltk sacrebleu

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import nltk
from collections import Counter
import sacrebleu
import numpy as np

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')  # Fixes LookupError for punkt_tab tokenizer

def tokenize(text):
    # Tokenize using nltk word_tokenize (depends on punkt and punkt_tab)
    try:
        return nltk.word_tokenize(text.lower())
    except LookupError:
        # Fallback simple whitespace tokenizer if nltk resources are missing
        return text.lower().split()

def compute_vocab(texts):
    vocab = Counter()
    for sent in texts:
        tokens = tokenize(sent)
        vocab.update(tokens)
    return vocab

def bleu_score(references, hypotheses):
    refs = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    return bleu.score

def eda_on_language(lang_code):
    print(f"\n{'='*40}\nStarting EDA for language: {lang_code}\n{'='*40}")

    # Load dataset for given language
    dataset = load_dataset("ai4bharat/samanantar", lang_code)
    df = pd.DataFrame(dataset['train'])

    # Drop missing values in src or tgt
    df = df.dropna(subset=['src', 'tgt']).reset_index(drop=True)

    # Sentence Length Ratio
    df['src_len'] = df['src'].apply(lambda x: len(tokenize(x)))
    df['tgt_len'] = df['tgt'].apply(lambda x: len(tokenize(x)))
    df['len_ratio'] = df['src_len'] / df['tgt_len'].replace(0, 1)

    print("1. Sentence Length Ratio stats:")
    print(df['len_ratio'].describe())

    plt.figure(figsize=(8,5))
    sns.histplot(df['len_ratio'], bins=50, kde=True, color='teal')
    plt.title(f"{lang_code}: Source-Target Sentence Length Ratio Distribution")
    plt.xlabel("Length Ratio (src_len / tgt_len)")
    plt.ylabel("Frequency")
    plt.show()

    # Vocabulary Overlap
    src_vocab = compute_vocab(df['src'])
    tgt_vocab = compute_vocab(df['tgt'])

    print(f"2. Vocabulary sizes:")
    print(f"  Source vocab size: {len(src_vocab)}")
    print(f"  Target vocab size: {len(tgt_vocab)}")

    overlap_tokens = set(src_vocab.keys()).intersection(set(tgt_vocab.keys()))
    print(f"  Overlapping tokens (exact match): {len(overlap_tokens)}")
    print(f"  Sample overlapping tokens: {list(overlap_tokens)[:20]}")

    # BLEU Score - sample up to 1000 rows for speed
    sample_df = df.sample(min(1000, len(df)), random_state=42).reset_index(drop=True)
    references = sample_df['tgt'].tolist()
    hypotheses = sample_df['src'].tolist()
    bleu = bleu_score(references, hypotheses)
    print(f"3. BLEU score (src as hypothesis, tgt as reference): {bleu:.2f}")

    # OOV words (outside top 10k vocab)
    top_n = 10000
    src_top_vocab = set([w for w, _ in src_vocab.most_common(top_n)])
    tgt_top_vocab = set([w for w, _ in tgt_vocab.most_common(top_n)])

    src_oov = set(src_vocab.keys()) - src_top_vocab
    tgt_oov = set(tgt_vocab.keys()) - tgt_top_vocab

    print(f"4. OOV words count (outside top {top_n}):")
    print(f"  Source OOV: {len(src_oov)}")
    print(f"  Target OOV: {len(tgt_oov)}")

    # Alignment pattern: sentence length correlation
    corr = df[['src_len', 'tgt_len']].corr().iloc[0,1]
    print(f"5. Sentence length correlation (Pearson r): {corr:.4f}")

    plt.figure(figsize=(8,6))
    sns.scatterplot(x='src_len', y='tgt_len', data=df.sample(min(1000, len(df)), random_state=42), alpha=0.4)
    plt.title(f"{lang_code}: Source vs Target Sentence Length")
    plt.xlabel("Source Sentence Length (words)")
    plt.ylabel("Target Sentence Length (words)")
    plt.show()

def main():
    languages = ['as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']

    for lang in languages:
        eda_on_language(lang)

if __name__ == "__main__":
    main()
