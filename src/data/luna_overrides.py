### New structures and functions to work with symmetry expanded fingerprints, override initial implementation


#from luna.mol.groups import AtomGroupPerceiver
#from luna.util.file import (is_directory_valid, new_unique_filename, remove_files)
#from luna.MyBio.PDB.PDBIO import PDBIO
#from luna.wrappers.base import MolWrapper
#from luna.wrappers.obabel import convert_molecule
#from luna.util.default_values import OPENBABEL
#from luna.mol.features import FeatureExtractor, ChemicalFeature
#from openbabel.pybel import readfile

#from luna.MyBio.selector import Selector, AtomSelector
#from luna.util.exceptions import MoleculeObjectError

#from luna.mol.atom import ExtendedAtom, AtomData


def load_all_structures(pdb_path, parser):
    structures = []
    pdb_block = []
    with open(pdb_path, 'r') as f:
        for line in f:
            pdb_block.append(line)
            if 'HEADER' in line: 
                struct_id = line.split()[1]
            if line.startswith('END'):
                try:
                    structure = parser.get_structure_from_pdb_block(struct_id, ''.join(pdb_block))
                except:
                    #TODO: this except should never trigger if PDB formatted correctly    
                    structure = parser.get_structure_from_pdb_block('prot', ''.join(pdb_block))
                structures.append(structure)
                pdb_block = []
    return structures

"""
def mic_save_to_file(models, output_file, select=None, write_conects=True,
                 write_end=True, preserve_atom_numbering=True, sort=False):
    io = PDBIO()
    with open(output_file, 'a+') as f:
        for i, model in enumerate(models):
            io.set_structure(model)
            write_end = i == len(models)-1
            io.save(f, select=select,
                    write_conects=False,
                    write_end=write_end,
                    preserve_atom_numbering=preserve_atom_numbering,
                    sort=sort)

def mic_biopython_entity_to_mol(models,
                            select=None,
                            amend_mol=True,
                            template=None,
                            add_h=False,
                            ph=None,
                            metals_coord=None,
                            wrapped=True,
                            openbabel=OPENBABEL,
                            tmp_path=None,
                            keep_tmp_files=False):
    

    tmp_path = tmp_path or tempfile.gettempdir()

    filename = new_unique_filename(tmp_path)
    pdb_file = '%s_pdb-file.pdb' % filename

    mic_save_to_file(models,
                 pdb_file,
                 select,
                 preserve_atom_numbering=True,
                 sort=True)
    
    ini_input_file = pdb_file
    if template is not None:
        if entity.level == "R" and entity.is_hetatm():
            # Note that the template molecule should have no explicit hydrogens
            # else the algorithm will fail.
            rdmol = read_mol_from_file(pdb_file, mol_format="pdb",
                                       removeHs=True)
            new_rdmol = template.assign_bond_order(rdmol, entity.resname)

            ini_input_file = '%s_tmp-mol-file.mol' % filename
            MolToMolFile(new_rdmol, ini_input_file)

            if not keep_tmp_files:
                remove_files([pdb_file])
        else:
            pass
    # Convert the PDB file to Mol file with the proper protonation
    # and hydrogen addition if required.
    mol_file = '%s_mol-file.mol' % filename
    ob_opt = {"error-level": 5}
    
    if add_h:
        if ph is not None:
            ob_opt["p"] = ph
        else:
            ob_opt["h"] = ""
    convert_molecule(ini_input_file, output_file=mol_file,
                     opts=ob_opt, openbabel=openbabel)


    # Currently, ignored atoms are only metals.
    ignored_atoms = []

    mol_obj = None
    try:
        # Create a new Mol object.
        mol_obj = next(readfile("mol", mol_file))
    except Exception:
        error_msg = ("An error occurred while parsing the file '%s' and "
                         "the molecule object could not be created. "
                         "Check the logs for more information."
                         % mol_file)
        raise MoleculeObjectError(error_msg)

    # Remove temporary files.
    if not keep_tmp_files:
        remove_files([ini_input_file, mol_file])

    if wrapped:
        mol_obj = MolWrapper(mol_obj)
    elif isinstance(mol_obj, MolWrapper):
        mol_obj = mol_obj.unwrap()

    return mol_obj, ignored_atoms


class MICPerceiver(AtomGroupPerceiver):
   
    def _fix_pharmacophoric_rules(self, atms_map):
        pass # DON'T consider metals when updating nearby properties - they may not be metals

    def _get_atoms(self, compound, keep_altloc = False):
        selector = Selector(keep_altloc=keep_altloc,
                            keep_hydrog=self.keep_hydrog)
        
        atoms = [atm for atm in compound.get_unpacked_list()
                if selector.accept_atom(atm)]
        
        return atoms    


    def add_dummy_groups(self, compounds):
        
        #This is specifically for adding dummy groups (waters, ions) to the atomic groups manager
        #These groups are given no properties and no atomic invariants (ie all 0s)

        for comp in sorted(compounds, key=lambda c: (c.parent.parent.id,
                                                   c.parent.id, c.idx)):

            old_atoms = self._get_atoms(comp, keep_altloc=True)
            
            if len(old_atoms) != 1:
                print(f"Warning: {len(atoms)} atoms found for {comp}, keeping first")

            new_atoms = [self._new_extended_atom(old_atoms[0])]
            new_atoms[0].invariants = [0, 0, 0, 0, 0, 0, 0]
            self.atm_grps_mngr.new_atm_grp(new_atoms,[ChemicalFeature("Atom")])

        return self.atm_grps_mngr

    def _get_mol_from_entity(self, compounds, metals_coord=None):

        atoms = [atm
                 for comp in sorted(compounds,
                                    key=lambda c: (c.parent.parent.id,
                                                   c.parent.id, c.idx))
                 for atm in self._get_atoms(comp)]

        model_lookup = set()
        models = [atom.get_parent_by_level('M') for atom in atoms]
        models = [m for m in models if m not in model_lookup and model_lookup.add(m) is None]

        atom_selector = AtomSelector(atoms, keep_altloc=False,
                                     keep_hydrog=self.keep_hydrog)

        mol_obj, ignored_atoms = \
            mic_biopython_entity_to_mol(models, atom_selector,
                                    amend_mol=self.amend_mol,
                                    add_h=self.add_h, ph=self.ph,
                                    metals_coord=metals_coord,
                                    tmp_path=self.tmp_path)

        # If the add_h property is set to False, the code will not remove any
        # existing hydrogens from the PDB structure. In these situations, the
        # list of atoms may contain hydrogens. But, we do not need to attribute
        # properties to hydrogens. We just need them to correctly set
        # properties to heavy atoms. So let's just ignore them.

        # MIC UPDATE: NEED TO REORDER TARGET ATOMS BASED ON THE ORDER WRITTEN TO FILE
        # OR COVALENT BONDS ARE INTERPRETED INCORRECTLY
   
        target_atoms = [] 
        for model in models:
            target_atoms += [atm for atm in atoms if atm.element != "H"
                        and atm not in ignored_atoms and atm.get_parent_by_level('M') == model]

        atm_obj_list = [atm for atm in mol_obj.get_atoms() if atm.get_atomic_num() != 1]
        for i in range(len(atm_obj_list)):
            assert target_atoms[i].element == atm_obj_list[i].get_symbol().upper()

        return mol_obj, target_atoms
"""
