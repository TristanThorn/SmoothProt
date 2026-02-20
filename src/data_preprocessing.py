import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. Safe translation of nucleotide sequence to protein
# ─────────────────────────────────────────────────────────────
def safe_translate(seq):
    CODON_TABLE = {
        'ATA':'I','ATC':'I','ATT':'I','ATG':'M',
        'ACA':'T','ACC':'T','ACG':'T','ACT':'T',
        'AAC':'N','AAT':'N','AAA':'K','AAG':'K',
        'AGC':'S','AGT':'S','AGA':'R','AGG':'R',
        'CTA':'L','CTC':'L','CTG':'L','CTT':'L',
        'CCA':'P','CCC':'P','CCG':'P','CCT':'P',
        'CAC':'H','CAT':'H','CAA':'Q','CAG':'Q',
        'CGA':'R','CGC':'R','CGG':'R','CGT':'R',
        'GTA':'V','GTC':'V','GTG':'V','GTT':'V',
        'GCA':'A','GCC':'A','GCG':'A','GCT':'A',
        'GAC':'D','GAT':'D','GAA':'E','GAG':'E',
        'GGA':'G','GGC':'G','GGG':'G','GGT':'G',
        'TCA':'S','TCC':'S','TCG':'S','TCT':'S',
        'TTC':'F','TTT':'F','TTA':'L','TTG':'L',
        'TAC':'Y','TAT':'Y','TAA':'*','TAG':'*',
        'TGC':'C','TGT':'C','TGA':'*','TGG':'W'
    }
    protein = []
    for i in range(0, len(seq) - 2, 3):
        aa = CODON_TABLE.get(seq[i:i + 3], 'X')
        if aa in ('*', 'X'):           # stop or unknown
            break
        protein.append(aa)
    return ''.join(protein)

# ─────────────────────────────────────────────────────────────
# 2.  Robust mutation applier
# ─────────────────────────────────────────────────────────────
def apply_mutations(wt_aa, mutations, verbose=False):
    """
    Convert a wild-type AA string into its mutant form given a
    colon or comma-separated string like:
        'SK50R:SI126F:SN142D'
    The dataset numbers residues after removing the initial Met,
    so we strip it from wt_aa first to keep indexes consistent.
    """
    if pd.isna(mutations) or mutations.strip() == '':
        return wt_aa           # nothing to do

    # dataset indexing excludes the initiator Met
    if wt_aa[0] == 'M':
        wt_aa = wt_aa[1:]

    seq      = list(wt_aa)
    delim    = ':' if ':' in mutations else ','
    for mut in mutations.split(delim):
        mut = mut.strip()
        if len(mut) < 4:                       # too short -> malformed
            if verbose:
                print(f"Skipping malformed mutation: {mut}")
            return None

        ref, pos_str, alt = mut[1], mut[2:-1], mut[-1]
        try:
            pos = int(pos_str) - 1             # 1‑based → 0‑based
        except ValueError:
            if verbose:
                print(f"Bad position in mutation: {mut}")
            return None

        if not (0 <= pos < len(seq)):          # out‑of‑range
            if verbose:
                print(f"Position {pos+1} outside sequence (len={len(seq)})")
            return None

        # apply even if the reference AA doesn't match exactly
        seq[pos] = alt
    return ''.join(seq)

# ─────────────────────────────────────────────────────────────
# 3.  Pre‑processing function (only the WT trimming line changed)
# ─────────────────────────────────────────────────────────────
def preprocess_gfp_data(fasta_path, tsv_path, sample_size=1000, out_csv=None):
    # — load WT nucleotide -> translate —
    with open(fasta_path) as f:
        wt_nt = ''.join(line.strip() for line in f if not line.startswith('>'))
    wt_aa = safe_translate(wt_nt)
    if wt_aa[0] == 'M':          # drop initiator Met for alignment
        wt_aa = wt_aa[1:]

    # — load dataset —
    df = pd.read_csv(tsv_path, sep='\t').dropna(subset=['medianBrightness'])
    df = df.sample(n=sample_size, random_state=42).copy()

    # — apply mutations —
    df['amino_acid_sequence'] = df['aaMutations'].apply(
        lambda m: apply_mutations(wt_aa, m)
    )
    df = df.dropna(subset=['amino_acid_sequence'])
    df = df[df['amino_acid_sequence'].str.len() == len(wt_aa)]

    # — binary mutation vector & counts —
    df['mutation_vector'] = df['amino_acid_sequence'].apply(
        lambda seq: [int(a != b) for a, b in zip(seq, wt_aa)]
    )
    df['mutation_count'] = df['mutation_vector'].apply(sum)

    final = df[['amino_acid_sequence', 'mutation_vector',
                'mutation_count', 'medianBrightness']].rename(
        columns={'medianBrightness': 'log_fluorescence'}
    )

    if out_csv:
        final.to_csv(out_csv, index=False)
    return final

if __name__ == "__main__":
    df = preprocess_gfp_data(
        fasta_path="../data/avGFP_reference_sequence.fa",
        tsv_path="../data/nucleotide_genotypes_to_brightness.tsv",
        sample_size=1000,
        out_csv="../output/processed_gfp_1000.csv"
    )
    print(df.head())