import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt

# from tqdm import tqdm

import os
import os.path as osp
import sys
import multiprocessing

import glob

#import luna
#from luna.MyBio.PDB.PDBParser import PDBParser
#from luna.mol.entry import Entry

from ..lunasmall.bioluna import PDBParser
from ..lunasmall.mol.entry import Entry
from .luna_utils import process_entry, sym_expand

class FPParser:
    def __init__(self, pdb_file, fp_type, 
            shell_number = 18, shell_radius= 0.25,
            count = True,
            length = 4096, blind = True, symexp_cutoff = 7, 
            entry_file = None, n_proc = 1, remove_files=False,
            session = None):

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
        self.session = session

        self.process()
            
    def parse_input_entries(self):
        entries_df = pd.read_csv(self.entry_file, names=['pdb_id', 'chain','name', 'resn','skip_symexp'], index_col = False, keep_default_na=False, na_values=[])
        entries_df['pdb_id'] = entries_df['pdb_id'].str.lower()
        self.symexp_pdbs = set(entries_df.loc[entries_df['skip_symexp'] != 'X']['pdb_id'])
        self.noexp_pdbs = set(entries_df.loc[entries_df['skip_symexp'] == 'X']['pdb_id'])
        
        entries_df['pdb_id'] = entries_df.apply(lambda x: x.pdb_id if x.skip_symexp == 'X' else f'{x.pdb_id}.symexp', axis=1)
        self._entries = entries_df[['pdb_id','chain','name','resn']].apply(lambda row: Entry(*row, is_hetatm=True, sep=':'), axis = 1).values
        print(self._entries)

    def _check_res(self, residue):
        return (residue.is_hetatm() or residue.is_metal() or residue.is_water()) and len(residue) == 1

    def _make_ion_entry(self, residue, pdb_id):
        return Entry(pdb_id=pdb_id, chain_id=residue.parent.id, comp_name=residue.resname, 
                 comp_num=residue.id[1], is_hetatm = True, sep=':')
    
    def parse_input_pdb(self):
        pdb_id =  osp.basename(self.pdb_file).replace('.pdb', '').lower()
        pdb_parser = PDBParser(PERMISSIVE=True, QUIET=True,
                                   FIX_EMPTY_CHAINS=True,
                                   FIX_ATOM_NAME_CONFLICT=True,
                                   FIX_OBABEL_FLAGS=False)

        structure = pdb_parser.get_structure('prot', osp.join(osp.dirname(self.pdb_file),pdb_id + '.pdb'))
        if self.symexp_cutoff > 0:
            pdb_id = osp.basename(self.pdb_file).replace('.pdb', '.symexp').lower()
        # self._entries = [self._make_ion_entry(res, pdb_id) for res in tqdm(structure.get_residues()) if self._check_res(res)]
        self._entries = [self._make_ion_entry(res, pdb_id) for res in structure.get_residues() if self._check_res(res)]


    def format_entries(self):
        return [str(e).split()[1].replace('>', '') for e in self.entries]

    def process(self):
        if self.entry_file: # symmetry expansion performed for each pdb within the jobs, no need to do it here
            self.parse_input_entries()

            symexp_cutoffs = [self.symexp_cutoff]*len(self.symexp_pdbs) + [-1]*len(self.noexp_pdbs)
            symexp_files = [sym_expand(osp.join(self.pdb_file, pdb + '.pdb'), co) for pdb, co in 
                # tqdm(zip(list(self.symexp_pdbs) + list(self.noexp_pdbs), symexp_cutoffs))]
                zip(list(self.symexp_pdbs) + list(self.noexp_pdbs), symexp_cutoffs)]


        else:
            symexp_files = [sym_expand(self.pdb_file, self.symexp_cutoff)]
            self.parse_input_pdb()
            
        prune = False if 'non' in self.fp_type else True
        
        #logging.getLogger().setLevel(logging.WARNING) 

        #with multiprocessing.Pool(processes=self.n_proc) as p:
        #    self._fps = list(tqdm(p.imap(self.multiprocess_func, self._entries), total=len(self._entries)))
        
        self._fps = []
        for entry in self._entries[:5]:
            self._fps.append(self.multiprocess_func(entry))


        if self.remove_files:
            for symexp_file in symexp_files:
                try:
                    os.remove(symexp_file)  # clean up
                except:
                    pass

    def multiprocess_func(self, entry):
        pdb_dir = osp.dirname(self.pdb_file) if self.pdb_file.endswith('.pdb') else self.pdb_file
        return process_entry(entry, not 'non' in self.fp_type, self.fp_type.split('-')[1], 
                self.shell_number, self.shell_radius, self.length, self.count, pdb_dir, self.blind, np.log1p, self.session)

    def get_as_dataframe(self):
        return pd.DataFrame(data=self.fps, index=self.format_entries(), columns=list(range(self.length)))

    @property
    def fps(self):
        return np.stack([fp[0] for fp in self._fps if fp[0] is not None])
    
    @property
    def entries(self):
        return [entry for entry, fp in zip(self._entries, self._fps) if fp[0] is not None]
   
    @property
    def errors(self):
        return [entry for entry, fp in zip(self._entries, self._fps) if fp[0] is None]
