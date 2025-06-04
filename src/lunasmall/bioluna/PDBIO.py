# PDBIO

import warnings

# Exceptions and Warnings
from Bio.PDB.PDBIO import PDBIO as BioPDBIO
from Bio.PDB.PDBIO import _select

class PDBException(Exception):
    """Define class PDBException."""

from .StructureBuilder import StructureBuilder

from Bio.Data.IUPACData import atom_weights

_ATOM_FORMAT_STRING = (
    "%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f%s%6.2f      %4s%2s%2s\n"
)
_PQR_ATOM_FORMAT_STRING = (
    "%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f %7s  %6s      %2s\n"
)

_TER_FORMAT_STRING = (
    "TER   %5i      %3s %c%4i%c                                                      \n"
)

_CONECT_FORMAT_STRING = "CONECT%5s%5s%5s%5s%5s\n"

class PDBIO(BioPDBIO):
    def set_structure(self, pdb_object):
        # MODBY: Alexandre Fassio
        # Get the Structure object.
        parent_struct = pdb_object.get_parent_by_level(level='S')
        # Save a copy of the CONECT records from structure.
        conects = parent_struct.conects

        # Check what the user is providing and build a structure appropriately
        if pdb_object.level == "S":
            structure = pdb_object
        else:
            sb = StructureBuilder()
            # MODBY: Alexandre
            # Keeps the id from the original structure.
            # MODBY: Alexandre
            # Now it pass the PDB file as a parameter
            sb.init_structure(parent_struct.id, parent_struct.pdb_file)
            sb.init_seg(' ')
            # Build parts as necessary
            if pdb_object.level == "M":
                sb.structure.add(pdb_object)
                self.structure = sb.structure
            else:
                sb.init_model(0)
                if pdb_object.level == "C":
                    sb.structure[0].add(pdb_object)
                else:
                    sb.init_chain('A')
                    if pdb_object.level == "R":
                        parent_id = pdb_object.parent.id
                        try:
                            sb.structure[0]['A'].id = parent_id
                        except Exception:
                            pass

                        # MODBY: Alexandre Fassio
                        # The old code (sb.structure[0]['A'].add(pdb_object)) used to try to access a chain whose id is always 'A'.
                        # However, when the informed residue has a different chain id, this access will raise a KeyError exception.
                        # Thus, we should always access the new chain id by using the parent_id variable.
                        sb.structure[0][parent_id].add(pdb_object)
                    else:
                        # Atom
                        sb.init_residue('DUM', ' ', 1, ' ')
                        parent_id = pdb_object.parent.parent.id
                        try:
                            sb.structure[0]['A'].id = parent_id
                        except Exception:
                            pass

                        # MODBY: Alexandre Fassio
                        # The old code (sb.structure[0]['A'].child_list[0].add(pdb_object)) used to try to access a chain whose id
                        # is always 'A'. However, when the informed atom has a different chain id, this access will raise a
                        # KeyError exception. Thus, we should always access the new chain id by using the parent_id variable.
                        sb.structure[0][parent_id].child_list[0].add(pdb_object)

            # Return structure
            structure = sb.structure

        structure.conects = conects
        self.structure = structure

    def save(self,
            file,
            select=_select, 
            write_end=True, 
            write_conects=True,
            preserve_atom_numbering=False):
        


        get_atom_line = self._get_atom_line

        if isinstance(file, str):
            fp = open(file, "w")
            close_file = 1
        else:
            # filehandle, I hope :-)
            fp = file
            close_file = 0

        valid_serial_numbers = set()
        serial_number_mapping = {}

        # multiple models?
        if len(self.structure) > 1 or self.use_model_flag:
            model_flag = 1
        else:
            model_flag = 0

        for model in self.structure.get_list():
            if not select.accept_model(model):
                continue
            # necessary for ENDMDL
            # do not write ENDMDL if no residues were written
            # for this model
            model_residues_written = 0
            if not preserve_atom_numbering:
                atom_number = 1
            if model_flag:
                fp.write("MODEL      %s\n" % model.serial_num)

            for chain in model.get_list():
                if not select.accept_chain(chain):
                    continue
                chain_id = chain.id
                if len(chain_id) > 1:
                    e = f"Chain id ('{chain_id}') exceeds PDB format limit."
                    raise PDBIOException(e)

                # necessary for TER
                # do not write TER if no residues were written
                # for this chain
                chain_residues_written = 0

                for residue in chain.get_unpacked_list():
                    if not select.accept_residue(residue):
                        continue
                    hetfield, resseq, icode = residue.id
                    resname = residue.resname
                    segid = residue.segid
                    resid = residue.id[1]
                    if resid > 9999:
                        e = f"Residue number ('{resid}') exceeds PDB format limit."
                        raise PDBIOException(e)

                    for atom in residue.get_unpacked_list():
                        if not select.accept_atom(atom):
                            continue
                        chain_residues_written = 1
                        model_residues_written = 1

                        if preserve_atom_numbering:
                            atom_number = atom.serial_number

                        serial_number_mapping[atom.get_serial_number()] = atom_number

                        try:
                            s = get_atom_line(
                                atom,
                                hetfield,
                                segid,
                                atom_number,
                                resname,
                                resseq,
                                icode,
                                chain_id,
                            )
                        except Exception as err:
                            # catch and re-raise with more information
                            raise PDBIOException(
                                f"Error when writing atom {atom.full_id}"
                            ) from err
                        else:
                            valid_serial_numbers.add(atom_number)
                            fp.write(s)
                            if not preserve_atom_numbering:
                                atom_number += 1

                if chain_residues_written:
                    fp.write(_TER_FORMAT_STRING % (atom_number, resname, chain_id, resseq, icode))

            if model_flag and model_residues_written:
                fp.write("ENDMDL\n")

                # MODBY: Alexandre Fassio
        # Print CONECT records
        if write_conects:
            conects = self.structure.conects
            for serial_number in sorted(conects):
                # Substitutes the old serial number by the new serial number
                new_serial_number = serial_number_mapping.get(serial_number, None)

                if new_serial_number not in valid_serial_numbers:
                    continue

                # It may return a list of lists as some atoms may have more than one CONECT line.
                bonded_atoms = conects[serial_number]
                max_num_fields = 4
                bonded_atoms_sets = [bonded_atoms[i:i + max_num_fields]
                                     for i in range(0, len(bonded_atoms),
                                                    max_num_fields)]

                for bonded_atoms in bonded_atoms_sets:
                    valid_bonded_atoms = []
                    for tmp_serial_number in bonded_atoms:
                        # Substitutes the old serial number by the new serial number
                        tmp_serial_number = serial_number_mapping.get(tmp_serial_number, None)

                        if tmp_serial_number in valid_serial_numbers:
                            valid_bonded_atoms.append(tmp_serial_number)

                    if len(valid_bonded_atoms) == 0:
                        continue

                    valid_bonded_atoms = [str(x) for x in valid_bonded_atoms]
                    missing_values = max_num_fields - len(valid_bonded_atoms)
                    valid_bonded_atoms += [''] * missing_values

                    record = _CONECT_FORMAT_STRING % (str(new_serial_number),
                                                      *valid_bonded_atoms)

                    fp.write(record)

        if write_end:
            fp.write('END\n')
        if close_file:
            fp.close()
