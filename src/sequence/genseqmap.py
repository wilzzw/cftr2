import os

from Bio import AlignIO

from database.query import get_mfa_id, protmodel2prot

# Read sequences from a multiple sequence alignment file
def sequences_from_file(seqfile, format='fasta'):
    seq_records = AlignIO.read(seqfile, format)
    # Extract sequence strings from mfa file
    extracted_seq = [s.seq for s in seq_records]
    return extracted_seq

# Generate sequence alignment mapping in terms of a dictionary
def genalign_map(mfa_filename):
    # Generate sequence alignment mapping in terms of dictionary
    aligned_sequences = sequences_from_file(mfa_filename)
    lengths_of_aligned_sequences = [len(seq) for seq in aligned_sequences]

    if len(lengths_of_aligned_sequences) != 2:
        raise NotImplementedError("Only two sequences are supported for now!")
    
    if len(set(lengths_of_aligned_sequences)) != 1:
        print("Aligned sequences don't have the same number of characters!")
        return

    resid1 = 0
    resid2 = 0
    align_map = {}
    for i in range(lengths_of_aligned_sequences[0]):
        rescode1 = aligned_sequences[0][i]
        rescode2 = aligned_sequences[1][i]
        
        if rescode1 != "-":
            resid1 += 1
        if rescode2 != "-":
            resid2 += 1
        if rescode1 != "-" and rescode2 != "-":
            align_map[resid1] = resid2

    return align_map

def protein_residue_alignment(protein_id1, protein_id2):
    if protein_id1 == protein_id2:
        # Load the protein sequence from the database; there is only one sequence in this case
        peptide_sequence = sequences_from_file(os.path.expanduser(f"~/cftr2/data/protein_seqs/{protein_id1}.fasta"))[0]
        resid_map = {r: r for r in range(1, len(peptide_sequence)+1)}
        return resid_map
    
    # Load the multiple sequence alignment file; query mfa_id from the database
    # By implementation, get_mfa_id returns a negative number if the alignment is reversed
    mfa_id = get_mfa_id(protein_id1, protein_id2)
    mfa_file = os.path.expanduser(f"~/cftr2/data/protseq_align/{abs(mfa_id)}.mfa")
    resid_map = genalign_map(mfa_file)

    if mfa_id < 0:
        resid_map = {r2: r1 for r1, r2 in resid_map.items()}
    return resid_map