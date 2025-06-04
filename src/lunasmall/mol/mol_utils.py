from os import remove
from os.path import exists

import string
import random

from ..bioluna.PDBIO import PDBIO
#from rdkit import Chem

from chimerax.core.commands import run

def new_unique_filename(path,
                        size=32,
                        chars=string.ascii_uppercase + string.digits,
                        retries=5):
    for r in range(retries):
        filename = '%s/%s' % (path, ('').join((random.choice(chars) for i in range(size))))
        if not exists(filename):
            return filename


def remove_files(files):
    for f in file:
        if exists(f):
            remove(f)

def mic_save_to_file(models, output_file, select=None, write_conects=True,
                 write_end=True, preserve_atom_numbering=True):
    io = PDBIO()
    with open(output_file, 'a+') as f:
        for i, model in enumerate(models):
            io.set_structure(model)
            write_end = i == len(models)-1
            io.save(f, select=select,
                    write_end=write_end,
                    write_conects = write_conects,
                    preserve_atom_numbering=preserve_atom_numbering)
    return output_file

def chimerax_biopython_entity_to_rdkitmol(models,
                            select=None,
                            tmp_path='/tmp',
                            keep_tmp_files=False,
                            session=None):

    filename = new_unique_filename(tmp_path)
    pdb_file = '%s_pdb-file.pdb' % filename

    mic_save_to_file(models,
                 pdb_file,
                 select,
                 preserve_atom_numbering=True)

    # use ChimeraX to load created PDB + add hydrogens; closest to obabel
    model_num = len(session.models) + 1

    chimerax_model = run(session, f"open {pdb_file} id #{model_num}")

    run(session, f"addh #{model_num}")
    run(session, f"save {pdb_file.replace('.pdb', '.mol2')} model #{model_num}")
    #run(session, f"close #27")

    rdkit_mol = Chem.MolFromMol2File(pdb_file.replace('.pdb', '.mol2'), 
            sanitize=False, removeHs = False)

    if rdkit_mol is None:
        return None

    rdkit_mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(rdkit_mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|\
                 Chem.SanitizeFlags.SANITIZE_KEKULIZE|\
                 Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|\
                 Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|\
                 Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|\
                 Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)

    return rdkit_mol
