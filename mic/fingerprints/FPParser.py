import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt

from tqdm import tqdm

import os
import os.path as osp
import sys
import multiprocessing

import glob

import luna
from luna.MyBio.PDB.PDBParser import PDBParser
from luna.mol.entry import Entry

from .luna_utils import process_entry, sym_expand

import logging
mic_logger = logging.getLogger('mic_logger')

class FPParser:
    def __init__(self, pdb_file, fp_type = 'prune-eifp', 
            shell_number = 18, shell_radius= 0.25,
            count = True,
            length = 4096, blind = True, symexp_cutoff = 7, 
            entry_file = None, n_proc = 1, remove_files=False,
            flag_distance=4, flag_threshold=3):

        self.pdb_file = pdb_file
        self.fp_type = fp_type
        self.entry_file = entry_file
        self.n_proc = n_proc
        self.blind = blind
        self.length = length    
        self.shell_number = shell_number
        self.shell_radius = shell_radius    
        self.count = count
        self.symexp_cutoff = symexp_cutoff
        self.remove_files = remove_files
        self.flag_distance = flag_distance
        self.flag_threshold = flag_threshold

        self.process()
            
    def parse_input_entries(self):
        entries_df = pd.read_csv(self.entry_file, names=['pdb_id', 'chain','name', 'resn','skip_symexp'], index_col = False, keep_default_na=False, na_values=[])
        self.symexp_pdbs = list(set(entries_df.loc[entries_df['skip_symexp'] != 'X']['pdb_id']))
        self.noexp_pdbs = list(set(entries_df.loc[entries_df['skip_symexp'] == 'X']['pdb_id']))
        
        self._entries = entries_df[['pdb_id','chain','name','resn']].apply(lambda row: Entry(*row, is_hetatm=True, sep=':'), axis = 1).values

    def _check_res(self, residue):
        return (residue.is_hetatm() or residue.is_metal() or residue.is_water()) and len(residue) == 1

    def _make_ion_entry(self, residue, pdb_id):
        return Entry(pdb_id=pdb_id, chain_id=residue.parent.id, comp_name=residue.resname, 
                 comp_num=residue.id[1], is_hetatm = True, sep=':')
    
    def parse_input_pdb(self):
        pdb_id =  osp.basename(self.pdb_file).replace('.pdb', '')
        pdb_parser = PDBParser(PERMISSIVE=True, QUIET=True,
                                   FIX_EMPTY_CHAINS=True,
                                   FIX_ATOM_NAME_CONFLICT=True,
                                   FIX_OBABEL_FLAGS=False)

        structure = pdb_parser.get_structure('prot', self.pdb_file)
        self._entries = [self._make_ion_entry(res, pdb_id) for res in tqdm(structure.get_residues()) if self._check_res(res)]

    def format_entries(self):
        return [str(e).split()[1].replace('>', '') for e in self.entries]

    def process(self):
        if self.entry_file:
            mic_logger.info(f'Processing entries...')
            self.parse_input_entries()

            symexp_cutoffs = [self.symexp_cutoff]*len(self.symexp_pdbs) + [-1]*len(self.noexp_pdbs)
            symexp_files = [sym_expand(osp.join(self.pdb_file, pdb + '.pdb'), co) for pdb, co in 
                tqdm(zip(list(self.symexp_pdbs) + list(self.noexp_pdbs), symexp_cutoffs))]
        
            symexp_map = dict(zip(self.symexp_pdbs+self.noexp_pdbs,
                                    [osp.basename(x).replace('.pdb', '') for x in symexp_files]))

            for entry in self._entries:
                entry._pdb_id = symexp_map[entry._pdb_id] # this could maybe be done above

            mic_logger.info(f'{len(self._entries)} entries detected in {len(symexp_files)} structures.')

        else:
            symexp_files = [sym_expand(self.pdb_file, self.symexp_cutoff)]
            self.pdb_file = symexp_files[0]
            self.parse_input_pdb()
            
        prune = False if 'non' in self.fp_type else True
        
        mic_logger.info(f'Calculating fingerprints for {len(self._entries)} sites...')
       
        logging.getLogger().setLevel(logging.WARNING) 
        with multiprocessing.Pool(processes=self.n_proc) as p:
            self._fps = list(tqdm(p.imap(self.multiprocess_func, self._entries), total=len(self._entries)))
        
        mic_logger.setLevel(logging.INFO)

        if self.remove_files:
            for symexp_file in symexp_files:
                try:
                    os.remove(symexp_file)  # clean up
                except:
                    mic_logger.warning(f"Unable to remove generated file {symexp_file}.")

    def multiprocess_func(self, entry):
        pdb_dir = osp.dirname(self.pdb_file) if self.pdb_file.endswith('.pdb') else self.pdb_file
        return process_entry(entry, not 'non' in self.fp_type, self.fp_type.split('-')[1], 
                self.shell_number, self.shell_radius, self.length, self.count, pdb_dir, self.blind, np.log1p,
                self.flag_distance, self.flag_threshold)

    def get_as_dataframe(self):
        fp_df = pd.DataFrame(data=self.fps, index=self.format_entries(), columns=list(range(self.length)))
        fp_df['flag_cluster'] = self.flags
        return fp_df

    @property
    def fps(self):
        return np.stack([fp[0] for fp in self._fps if fp[0] is not None])
    
    @property
    def entries(self):
        return [entry for entry, fp in zip(self._entries, self._fps) if fp[0] is not None]
  
    @property
    def flags(self):
        return np.array([fp[2] for fp in self._fps if fp[0] is not None])
 
    @property
    def errors(self):
        return [(entry, fp[1]) for entry, fp in zip(self._entries, self._fps) if fp[0] is None]

def turn_site_id_into_label(string, labels):
    return labels.index(string)

def convert_to_fp(indeces, counts, norm = True):
    fp  = np.zeros(4096)
    
    if type(indeces) == str:
        for idx, count in zip([int(x) for x in indeces.split('\t')],
                            [int(x) for x in counts.split('\t')]):
            fp[idx] = count
            
    elif type(indeces) == int: # corner case - single element fingerprints
        assert type(counts) == int
        fp[indeces] = counts
        
    return np.log1p(fp) if norm else fp # NEW!

def load_fps(file_matches, file_dir = "", norm = True):
    run_dfs = []
    for file_match in file_matches:
        for file in glob.glob(f'{file_dir}/{file_match}'):
            run_dfs += [pd.read_csv(file, names= ["entry", "indices", "counts"])]
                
    combined_df = pd.concat(run_dfs, ignore_index = True)
    
    # post processin - fps, labels
    combined_df['fp'] = combined_df.apply(lambda x: convert_to_fp(x.indices, x.counts, norm = norm), axis = 1)
    combined_df['label'] = combined_df['entry'].apply(lambda x: x.split(':')[2])
    return combined_df

def get_id_splits(path_to_ids):
    split_ids = {}
    id_files = glob.glob(path_to_ids)
    for f in id_files:
        name = f.split('_')[-1].replace('.npy', '')
        split_ids[name] = np.load(f, allow_pickle=True)
        split_ids[name] = list(set(split_ids[name]))
    return split_ids

def prep_data_dfs(fp_names, fp_path, labels, norm = True, id_splits = True, id_path = None, split_fp = True):
    df = load_fps(fp_names, fp_path, norm = norm)
    fp = pd.DataFrame(np.stack(df['fp']), columns = np.arange(4096))
    entry = pd.DataFrame(np.stack(df['entry']), columns = ["entry_id"])
    label = pd.DataFrame(np.stack(df['label']), columns = ["site_id"])
    join1 = pd.merge(entry,fp,left_index=True, right_index=True)
    df = pd.merge(join1,label,left_index=True, right_index=True)
    df['label'] = df['site_id'].apply(turn_site_id_into_label, args=(labels,)) #convert to labels
    df.set_index('entry_id', inplace=True)
    if id_splits:
        id_splits = get_id_splits(id_path)
        return df, id_splits
    return df


def string_to_array(string, norm = True):
    tokens = string.split(',')
    if norm:
        return np.log1p(np.array([int(t) for t in tokens]))
    else:
        return np.array([int(t) for t in tokens])

def site_id_to_label(string, labels = ['HOH', 'NA', 'K', 'MG', 'CA', 'CL', 'IOD', 'BR', 'ZN', 'FE', 'MN']):
    return labels.index(string)

def load_fp_csv(csv_path, norm = True):
    from_file = pd.read_csv(csv_path, index_col = 0)
    from_file['entry_id'] = from_file['entry'].apply(lambda x: x.split()[1].replace('>', ''))
    from_file['np_fps'] = from_file['fps'].apply(string_to_array, args=(norm,))
    from_file['site_id'] = from_file['entry'].apply(lambda x: x.split(':')[3])
    from_file['label'] = from_file['site_id'].apply(site_id_to_label)
    return from_file

