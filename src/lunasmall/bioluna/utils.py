def get_entity_from_entry(entity, entry, model=0):
    structure = entity.get_parent_by_level("S")
    model = structure[model]

    if entry.chain_id in model.child_dict:
        chain = model[entry.chain_id]

        if entry.comp_name is not None and entry.comp_num is not None:
            ligand_key = entry.get_biopython_key()
            
            if ligand_key in chain.child_dict:
                target_entity = chain[ligand_key]
            else:
                error_msg = ("Ligand '%s' does not exist in PDB '%s'."
                             % (entry.to_string(':'),
                                structure.get_id()))
                raise MoleculeNotFoundError(error_msg)
        else:
            target_entity = chain
    else:
        error_msg = ("The informed chain id '%s' for the ligand entry '%s' "
                     "does not exist in the PDB '%s'."
                     % (entry.chain_id, entry.to_string(':'),
                        structure.get_id()))
        raise ValueError(error_msg)

    return target_entity

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