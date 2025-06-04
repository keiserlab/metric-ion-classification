from Bio.PDB.Entity import Entity as BioEntity
from Bio.PDB.Model import Model as BioModel
from Bio.PDB.Atom import Atom as BioAtom
from Bio.PDB.Residue import Residue as BioResidue
from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.Structure import Structure as BioStructure

from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.Entity import Entity, DisorderedEntityWrapper

from Bio.PDB.Polypeptide import is_aa

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain

_atom_name_dict = {}
_atom_name_dict["N"] = 1
_atom_name_dict["CA"] = 2
_atom_name_dict["C"] = 3
_atom_name_dict["O"] = 4

METALS = ["LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "V", "CR",
          "MN", "MN3", "FE2", "FE", "CO", "3CO", "NI", "3NI", "CU1", "CU",
          "CU3", "ZN", "AL", "GA", "ZR", "MO", "4MO", "6MO", "RU", "RH",
          "RH3", "PD", "AG", "CD", "IN", "W", "RE", "OS", "OS4", "IR",
          "IR3", "PT", "PT4", "AU", "AU3", "HG", "TL", "PB", "BS3", "0BE",
          "4TI", "Y1", "YT3", "TA0"]

class Entity(BioEntity):
    def get_parent_by_level(self, level):
        """Return the parent Entity object in the specified level."""
        opts = ("A", "R", "C", "M", "S")

        if level not in opts:
            raise ValueError("Level must be one of the following options: A, R, C, M, or S.")

        if opts.index(level) < opts.index(self.level):
            raise ValueError("Parent object cannot be recovered. "
                             "Informed level ('%s') is not above the current object level ('%s'). "
                             "The correct hierarchy is: A < R < C < M < S."
                             % (level, self.level))

        if self.level == level:
            return self
        else:
            return self.parent.get_parent_by_level(level)

class Atom(BioAtom): # Note: Biopython Atom does NOT subclass Entity
    def get_parent_by_level(self, level):
        """Return the parent Entity object in the specified level."""
        if self.level == level:
            return self
        else:
            return self.parent.get_parent_by_level(level)

class Chain(BioChain, Entity):
    def set_as_target(self, is_target=True):
        self._is_target = is_target

        for r in self.get_residues():
            r.set_as_target(is_target)

class Residue(BioResidue, Entity):
    def __init__(self, id, resname, segid, idx, at_line=None):
        super().__init__(id, resname, segid) # resini

        self.idx = idx
        self.at_line = at_line
        self._is_target = False
        self.cluster_id = None


    # additional functions from Alexandre
    def __lt__(self, r2):
        return self.idx < r2.idx

    def _sort(self, a1, a2):
        """Sort the Atom objects.

        Atoms are sorted alphabetically according to their name,
        but N, CA, C, O always come first.

        Arguments:
         - a1, a2 - Atom objects

        """
        name1 = a1.get_name()
        name2 = a2.get_name()
        if name1 == name2:
            return(cmp(a1.get_altloc(), a2.get_altloc()))
        if name1 in _atom_name_dict:
            index1 = _atom_name_dict[name1]
        else:
            index1 = None
        if name2 in _atom_name_dict:
            index2 = _atom_name_dict[name2]
        else:
            index2 = None
        if index1 and index2:
            return cmp(index1, index2)
        if index1:
            return -1
        if index2:
            return 1

        return cmp(name1, name2)

    @property
    def metal_coordination(self):
        metal_coordination = {}
        for a in self.get_atoms():
            if a.has_metal_coordination():
                metal_coordination[a] = a.metal_coordination
        return metal_coordination

    def set_as_target(self, is_target=True):
        self._is_target = is_target

    def is_target(self):
        return self._is_target

    def is_nucleotide(self):
        return self.get_id()[0] == " " and not self.is_residue()

    def is_water(self):
        """Return 1 if the residue is a water molecule."""
        return self.get_id()[0] == "W"

    def is_metal(self):
        """Return True if the residue is a metal."""
        return (self.get_id()[0].startswith("H_")
                and self.resname in METALS)
    def is_hetatm(self):
        """Return True if the residue is an hetero group."""
        return (self.get_id()[0].startswith("H_")
                and not self.is_metal())

    def is_residue(self):
        """Return True if the residue is an amino acid."""
        return self.get_id()[0] == " " and is_aa(self.resname)

    def get_class(self):
        if self.is_water():
            return "Water"
        if self.is_hetatm():
            return "Hetatm"
        if self.is_residue():
            return "Residue"
        if self.is_nucleotide():
            return "Nucleotide"
        return "Unknown"


class Model(BioModel, Entity):
    def get_atoms(self):
        for r in self.get_residues():
            for a in r.get_unpacked_list():
                yield a

class Structure(BioStructure, Entity):

    def __init__(self, id, pdb_file=None):
        super().__init__(id)
        self.pdb_file = pdb_file
