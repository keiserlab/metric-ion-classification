
import numpy as np
from collections import defaultdict
from itertools import chain
from ..bioluna.selector import Selector, AtomSelector
from ..interaction.features import ChemicalFeature
from .mol_utils import chimerax_biopython_entity_to_rdkitmol
from .atoms import ExtendedAtom, AtomData, ChimeraXAtom

def atom_coordinates(atoms):
    return np.array([x.coord for x in atoms])
def centroid(arr, decimals=3):
    return np.around(np.mean(arr, axis=0), decimals)
def euclidean_distance(p1, p2, decimals=3):
    return round(np.linalg.norm(p1-p2), decimals)

class AtomGroupPerceiver():
    def __init__(self, feature_extractor=None, 
                expand_selection=True,
                tmp_path=None):

        self.feature_extractor = feature_extractor
        self.expand_selection = expand_selection
        self.tmp_path = tmp_path
        self.keep_hydrog = True

    def add_dummy_groups(self, compounds, target = False):
        """
        This is specifically for adding dummy groups (waters, other ions) to the atomic groups manager
        These groups are given no properties and no atomic invariants (all 0s)
        """

        trgt_grp = None

        for cpd in compounds:
            old_atoms = cpd.atoms 

            if len(old_atoms) != 1:
                print(f"Warning: {len(old_atoms)} atoms found for {cpd}, keeping first")

            new_atoms = [self._new_extended_atom(old_atoms[0])]
            new_atoms[0].invariants = [0, 0, 0, 0, 0, 0, 0]

            if target:
                trgt_grp = AtomGroup(new_atoms, [ChemicalFeature("Atom")])
            else:
                self.atm_grps_mngr.new_atm_grp(new_atoms,[ChemicalFeature("Atom")])

        return trgt_grp, self.atm_grps_mngr
    
    def perceive_atom_groups(self, compounds, session=None):

        self.atm_grps_mngr = AtomGroupsManager()
        self.atm_mapping = {}
        compounds = set(compounds)
        self._assign_properties(compounds, session=session)
        return self.atm_grps_mngr

    def _new_extended_atom(self, atm, invariants=None):
        if atm not in self.atm_mapping:
            self.atm_mapping[atm] = ExtendedAtom(atm, invariants=invariants)

        else:
            print("Atm already in self.atm_mapping", atm)

        return self.atm_mapping[atm]

    def _assign_invariants(self, atom):
        total_degree = atom.num_bonds

        cx_valence = atom.element.valence
        ob_valence = cx_valence if cx_valence <= 4 else (8 - cx_valence if atom.element.name != "P" else 5)

        heavy_neighbors = len([n_atm for n_atm in atom.neighbors if n_atm.element.number != 1])
        n_h_neighbors = len(atom.neighbors) - heavy_neighbors

        in_ring = len(atom.rings()) > 0

        if '-' in atom.idatm_type:
            charge = -1
        elif '+' in atom.idatm_type:
            charge = 1
        else:
            charge = 0

        print([atom.name, atom.element.name, heavy_neighbors, 
                atom.element.valence - n_h_neighbors, 
                atom.element.number,
                round(atom.element.mass),
                charge,
                n_h_neighbors, int(in_ring)])
        return [heavy_neighbors, 
                atom.element.valence - n_h_neighbors, 
                atom.element.number,
                round(atom.element.mass),
                charge,
                n_h_neighbors, int(in_ring)]

        #return [atom.GetTotalDegree(),
        #        ((atom.GetExplicitValence() + atom.GetImplicitValence()) - atom.GetTotalNumHs(includeNeighbors=True)),
        #        atom.GetAtomicNum(),
        #        int(round(atom.GetMass())),
        #        atom.GetFormalCharge(),
        #        atom.GetTotalNumHs(includeNeighbors=True),
        #        int(atom.IsInRing())]

    def _assign_properties(self, trgt_compounds, session=None):
        """
        Assigns properties to target compounds and builds atom groups.
        Tracks hydrogen addition and removal explicitly.
        """
        atm_obj_list = [atm for cpd in trgt_compounds for atm in cpd.atoms if atm.element.number != 1]
        trgt_atms = {}
        for i, atm in enumerate(atm_obj_list):
            atm_key = (atm.residue.name, atm.residue.number, atm.name)
            trgt_atms[atm_key] = self._new_extended_atom(atm, invariants=self._assign_invariants(atm))
        
        for atm in atm_obj_list:
            atm_key = (atm.residue.name, atm.residue.number, atm.name)
            for nb_atm in atm.neighbors:
                if nb_atm.element.number == 1:  
                    # ignore H-involved interation
                    continue
                nb_key = (nb_atm.residue.name, nb_atm.residue.number, nb_atm.name)
                atom_info = AtomData(nb_atm.element.number, nb_atm.coord, nb_key)
                trgt_atms[atm_key]._nb_info.append(atom_info)
        session.logger.info(f"trgt_atms: {trgt_atms}")

        for atm_key in trgt_atms:
            self.atm_grps_mngr.new_atm_grp([trgt_atms[atm_key]], [ChemicalFeature("Atom")])
        return

    def _get_mol_from_entity(self, compounds, session=None):

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
        rdkit_mol = chimerax_biopython_entity_to_rdkitmol(models, atom_selector,
                                    tmp_path=self.tmp_path,
                                    session=session)
        if rdkit_mol is None:
            return None, None

        target_atoms = [atm for atm in atoms if atm.element != "H"]
        atm_obj_list = [atm for atm in rdkit_mol.GetAtoms() if atm.GetAtomicNum() != 1]

        for i in range(len(atm_obj_list)):
            assert target_atoms[i].element == atm_obj_list[i].GetSymbol().upper()

        return rdkit_mol, target_atoms
    
    def _get_atoms(self, compound, keep_altloc = False):
        selector = Selector(keep_altloc=keep_altloc,
                            keep_hydrog=self.keep_hydrog)
        
        atoms = [atm for atm in compound.get_unpacked_list()
                if selector.accept_atom(atm)]
        
        return atoms 



import time
        
class AtomGroupsManager():
    def __init__(self, atm_grps=None, entry=None):
        self._atm_grps = []
        self.entry = entry
        self._child_dict = {}
        self._compounds = set()

        self.add_atm_grps(atm_grps)

    @property
    def atm_grps(self):
        return self._atm_grps

    @property
    def compounds(self):
        return self._compounds

    @property
    def child_dict(self):
        return self._child_dict

    @property
    def size(self):
        """int, read-only: The number of atom groups in ``atm_grps``."""
        return len(self._atm_grps)

    @property
    def summary(self):
        """dict, read-only: The number of physicochemical features in
        ``atm_grps``."""
        summary = defaultdict(int)
        for grp in self.atm_grps:
            for feature in grp.features:
                summary[feature] += 1

        return summary

    def find_atm_grp(self, atoms):
        return self.child_dict.get(tuple(sorted(atoms)), None)

    def get_all_interactions(self):
        return set(chain.from_iterable([atm_grp.interactions
                                        for atm_grp in self.atm_grps]))

    def apply_filter(self, func):
        for atm_grp in self.atm_grps:
            if func(atm_grp):
                yield atm_grp


    def add_atm_grps(self, atm_grps):
        atm_grps = atm_grps or []

        self._atm_grps = list(set(self._atm_grps + list(atm_grps)))

        for atm_grp in atm_grps:
            self.child_dict[tuple(sorted(atm_grp.atoms))] = atm_grp
            self._compounds.update(atm_grp.compounds)
            atm_grp.manager = self

    def remove_atm_grps(self, atm_grps):
        self._atm_grps = list(set(self._atm_grps) - set(atm_grps))

        for atm_grp in atm_grps:
            atm_grp.clear_refs()

            # Remove the atom group from the dict.
            key = tuple(sorted(atm_grp.atoms))
            if key in self.child_dict:
                del self.child_dict[key]

    def new_atm_grp(self, atoms, features=None, interactions=None):
        key = tuple(sorted(atoms))
        features = features or []
        interactions = interactions or []

        if key in self.child_dict:
            atm_grp = self.child_dict[key]
        else:
            atm_grp = AtomGroup(atoms)
            self.add_atm_grps([atm_grp])

        if features:
            atm_grp.add_features(features)

        if interactions:
            atm_grp.add_interactions(interactions)

        return atm_grp

    def merge_hydrophobic_atoms(self, interactions_mngr):
        # Only hydrophobic atom groups.
        hydrop_atm_grps = list(self.filter_by_types(["Hydrophobic"]))

        # Hydrophobic islands dictionary. Keys are integer values and items are
        # defined by a set of atom groups.
        hydrop_islands = defaultdict(set)

        # It stores a mapping of an atom (represented by its full id) and a
        # hydrophobic island (defined by its keys).
        atm_mapping = {}

        island_id = 0
        for atm_grp in hydrop_atm_grps:
            # Hydrophobic atoms are defined always as only one atom.
            atm = atm_grp.atoms[0]

            # Recover the groups of all neighbors of this atom (it will merge
            # all existing islands).
            nb_grps = set([atm_mapping[nbi.full_id]
                           for nbi in atm.neighbors_info
                           if nbi.full_id in atm_mapping])

            # Already there are hydrophobic islands formed by the neighbors of
            # this atom.
            if nb_grps:
                # Merge all groups of the neighbors of this atom.
                new_island = \
                    set(chain.from_iterable([hydrop_islands.pop(nb_grp_id)
                                             for nb_grp_id in nb_grps]))
                # Include this atom to the merged group.
                new_island.add(atm)

                for k in atm_mapping:
                    if atm_mapping[k] in nb_grps:
                        atm_mapping[k] = island_id

                hydrop_islands[island_id] = new_island
                atm_mapping[atm.get_full_id()] = island_id
            else:
                atm_mapping[atm.get_full_id()] = island_id
                hydrop_islands[island_id].add(atm)

            island_id += 1

        # Create AtomGroup objects for the hydrophobic islands
        for island_id in hydrop_islands:
            # It will update an existing atom group or create a new one
            # with the informed parameters.
            hydrophobe = self.new_atm_grp(hydrop_islands[island_id],
                                          [ChemicalFeature("Hydrophobe")])
            # Update the island information
            hydrop_islands[island_id] = hydrophobe

        hydrop_interactions = \
            list(interactions_mngr.filter_by_types(["Hydrophobic"]))
        island_island_inter = defaultdict(set)
        for inter in hydrop_interactions:
            src_atm = inter.src_grp.atoms[0]
            trgt_atm = inter.trgt_grp.atoms[0]

            # The two island ids are used as key.
            key = tuple(sorted([atm_mapping[src_atm.get_full_id()],
                                atm_mapping[trgt_atm.get_full_id()]]))

            island_island_inter[key].add(inter)

        interactions = set()
        for k in island_island_inter:
            island_atms = defaultdict(set)
            for inter in island_island_inter[k]:
                src_atm = inter.src_grp.atoms[0]
                trgt_atm = inter.trgt_grp.atoms[0]

                island_atms[atm_mapping[src_atm.get_full_id()]].add(src_atm)
                island_atms[atm_mapping[trgt_atm.get_full_id()]].add(trgt_atm)

            centroid1 = centroid(atom_coordinates(island_atms[k[0]]))
            centroid2 = centroid(atom_coordinates(island_atms[k[1]]))
            cc_dist = euclidean_distance(centroid1, centroid2)

            params = {"dist_hydrop_inter": cc_dist}

            inter = InteractionType(hydrop_islands[k[0]],
                                    hydrop_islands[k[1]],
                                    "Hydrophobic",
                                    src_interacting_atms=island_atms[k[0]],
                                    trgt_interacting_atms=island_atms[k[1]],
                                    params=params)
            interactions.add(inter)

        interactions_mngr.add_interactions(interactions)
        # Remove atom-atom hydrophobic interactions.
        interactions_mngr.remove_interactions(hydrop_interactions)

        for atm_grp in hydrop_atm_grps:
            features = [f for f in atm_grp.features if f.name != "Hydrophobic"]
            atm_grp.features = features


    def __len__(self):
        # Number of atom groups.
        return self.size

    def __iter__(self):
        """Iterate over children."""
        for atm_grp in self.atm_grps:
            yield atm_grp
    
    def filter_by_types(self, types, must_contain_all=True):
        for atm_grp in self.atm_grps:
            if must_contain_all:
                if set(types).issubset(set(atm_grp.feature_names)):
                    yield atm_grp
            else:
                if len(set(types) & set(atm_grp.feature_names)) > 0:
                    yield atm_grp


class AtomGroup():
    def __init__(self,
                 atoms,
                 features=None,
                 interactions=None,
                 recursive=False,
                 manager=None):
        self._atoms = sorted(atoms)

        # Atom properties
        self._coords = atom_coordinates([atom._atom for atom in atoms])
        self._centroid = centroid(self._coords)

        features = features or []
        self._features = sorted(features)

        self._interactions = interactions or []
        self._hash_cache = None

        self._manager = manager

        self._recursive = recursive

        if recursive:
            for atm in self.atoms:
                atm.add_atm_grps([self])

    @property
    def atoms(self):
        """iterable of :class:`~luna.mol.atom.ExtendedAtom`, read-only: \
            The sequence of atoms that belong to an atom group."""
        return self._atoms

    @property
    def compounds(self):
        """set of :class:`~luna.MyBio.PDB.Residue.Residue`, read-only: \
            The set of unique compounds that contain the atoms in ``atoms``.

        As an atom group can be formed by the union of two or more compounds
        (e.g., amide of peptide bonds), it may return more than one compound.
        """
        return set([a.parent for a in self._atoms])

    @property
    def coords(self):
        """ array-like of floats : Atomic coordinates (x, y, z) of each \
        atom in ``atoms``."""
        return self._coords

    @property
    def centroid(self):
        """ array-like of floats, read-only: The centroid (x, y, z) of the \
        atom group.

        If ``atoms`` contains only one atom, then ``centroid`` returns the same
        as ``coords``.
        """
        return self._centroid

    @property
    def features(self):
        """iterable of :class:`~luna.mol.features.ChemicalFeature`: \
                A sequence of chemical features.

        To add or remove a feature use :py:meth:`add_features`
        or :py:meth:`remove_features`, respectively."""
        return self._features

    @features.setter
    def features(self, features):
        self._features = sorted(features)
        # Reset hash.
        self._hash_cache = None

    @property
    def feature_names(self):
        """iterable of str: The name of each chemical feature in \
        ``features``."""
        return [f.name for f in self.features]

    @property
    def interactions(self):
        """iterable of :class:`~luna.interaction.type.InteractionType`: \
            The sequence of interactions established by an atom group.

        To add or remove an interaction use :py:meth:`add_interactions`
        or :py:meth:`remove_interactions`, respectively."""
        return self._interactions

    @interactions.setter
    def interactions(self, interactions):
        self._interactions = interactions

    @property
    def manager(self):
        """`AtomGroupsManager`: The `AtomGroupsManager` object that contains \
        an `AtomGroup` object."""
        return self._manager

    @manager.setter
    def manager(self, manager):
        if isinstance(manager, AtomGroupsManager):
            self._manager = manager
        else:
            raise IllegalArgumentError("The informed atom group manager must "
                                       "be an instance of '%s'."
                                       % AtomGroupsManager)

    @property
    def size(self):
        """int: The number of atoms comprising an atom group."""
        return len(self.atoms)

    def has_atom(self, atom):
        """Check if an atom group contains a given atom ``atom``.

        Parameters
        ----------
        atom : :class:`~luna.mol.atom.ExtendedAtom`

        Returns
        -------
         : bool
            If the atom group contains or not ``atom``.
        """
        return atom in self.atoms

    def contain_group(self, atm_grp):
        """Check if the atom group ``atm_grp`` is a subset of this atom group.

        For example, consider the benzene molecule.
        Its aromatic ring itself forms an `AtomGroup` object composed of all of
        its six atoms. Consider now any subset of carbons in the benzene
        molecule. This subset forms an `AtomGroup` object that is part of the
        group formed by the aromatic ring. Therefore, in this example,
        :meth:`contain_group` will return True because the aromatic ring
        contains the subset of hydrophobic atoms.

        Parameters
        ----------
        atm_grp : :class:`~luna.mol.groups.AtomGroup`

        Returns
        -------
         : bool
            If one atom group contains another atom group.
        """
        return set(atm_grp.atoms).issubset(set(self.atoms))

    def get_serial_numbers(self):
        """Get the serial number of each atom in an atom group."""
        return [a.get_serial_number() for a in self.atoms]

    def get_chains(self):
        """Get all unique chains in an atom group."""
        return sorted(set([a.get_parent_by_level("C").id for a in self.atoms]))

    def get_interactions_with(self, atm_grp):
        """Get all interactions that an atom group establishes with another
        atom group ``atm_grp``.

        Returns
        -------
         : iterable of :class:`~luna.interactions.type.InteractionType`
           All interactions established with the atom group ``atm_grp``.
        """
        target_interactions = []

        for inter in self.interactions:
            if inter.src_grp == atm_grp or inter.trgt_grp == atm_grp:
                target_interactions.append(inter)
        return target_interactions

    def get_shortest_path_length(self, trgt_grp, cutoff=None):
        """Compute the shortest path length between this atom group to another
        atom group ``trgt_grp``.

        The shortest path between two atom groups is defined as the shortest
        path between any of their atoms, which are calculated using
        Dijkstra’s algorithm.

        If ``manager`` is not provided, None is returned.

        If there is not any path between ``src_grp`` and ``trgt_grp``,
        infinite is returned.

        Parameters
        ----------
        trgt_grp : `AtomGroup`
            The target atom group to calculate the shortest path.
        cutoff : int, optional
            Only paths of length <= ``cutoff`` are returned.
            If None, all path lengths are considered.

        Returns
        -------
         : int, float('inf'), or None:
            The shortest path.
        """
        if self.manager is not None:
            return self.manager.get_shortest_path_length(self,
                                                         trgt_grp,
                                                         cutoff)
        return None

    def add_features(self, features):
        """ Add :class:`~luna.mol.features.ChemicalFeature` objects
        to ``features``."""
        self._features = sorted(set(self.features + list(features)))
        # Reset hash.
        self._hash_cache = None

    def remove_features(self, features):
        """ Remove :class:`~luna.mol.features.ChemicalFeature` objects
        from ``features``."""
        self._features = sorted(set(self.features) - set(features))
        # Reset hash.
        self._hash_cache = None

    def add_interactions(self, interactions):
        """ Add :class:`~luna.interaction.type.InteractionType` objects
        to ``interactions``."""
        self._interactions = list(set(self.interactions + list(interactions)))

    def remove_interactions(self, interactions):
        """ Remove :class:`~luna.interaction.type.InteractionType` objects
        from ``interactions``."""
        self._interactions = list(set(self.interactions) - set(interactions))

    def is_water(self):
        """Return True if all atoms in the atom group belong to water
        molecules."""
        return all([a.parent.is_water() for a in self.atoms])

    def is_hetatm(self):
        """Return True if all atoms in the atom group belong to hetero group,
        i.e., non-standard residues of proteins, DNAs, or RNAs, as well as
        atoms in other kinds of groups, such as carbohydrates, substrates,
        ligands, solvent, and metal ions.

        Hetero groups are designated by the flag HETATM in the PDB format."""
        return all([a.parent.is_hetatm() for a in self.atoms])

    def is_metal(self):
        """Return True if all atoms in the atom group are metal ions."""
        return all([a.parent.is_metal() for a in self.atoms])

    def is_residue(self):
        """Return True if all atoms in the atom group belong to standard
        residues of proteins."""
        return all([a.parent.is_residue() for a in self.atoms])

    def is_nucleotide(self):
        """Return True if all atoms in the atom group belong to nucleotides."""
        return all([a.parent.is_nucleotide() for a in self.atoms])

    def is_mixed(self):
        """Return True if the atoms in the atom group belong to different
        compound classes (water, hetero group, residue, or nucleotide)."""
        return len(set([a.parent.get_class() for a in self.atoms])) > 1

    def has_water(self):
        """Return True if at least one atom in the atom group belongs to a
        water molecule."""
        return any([a.parent.is_water() for a in self.atoms])

    def has_hetatm(self):
        """Return True if at least one atom in the atom group belongs to a
        hetero group, i.e., non-standard residues of proteins, DNAs, or RNAs,
        as well as atoms in other kinds of groups, such as carbohydrates,
        substrates, ligands, solvent, and metal ions."""
        return any([a.parent.is_hetatm() for a in self.atoms])

    def has_metal(self):
        """Return True if at least one atom in the atom group is a metal."""
        return any([a.parent.is_metal() for a in self.atoms])

    def has_residue(self):
        """Return True if at least one atom in the atom group belongs to a
        standard residue of proteins."""
        return any([a.parent.is_residue() for a in self.atoms])

    def has_nucleotide(self):
        """Return True if at least one atom in the atom group belongs to a
        nucleotide."""
        return any([a.parent.is_nucleotide() for a in self.atoms])

    def has_target(self):
        """Return True if at least one compound is the target of LUNA's
        analysis"""
        return any([a.parent.is_target() for a in self.atoms])

    def clear_refs(self):
        """References to this `AtomGroup` instance will be removed from the
        list of atom groups of each atom in ``atoms``."""
        if self._recursive:
            for atm in self.atoms:
                atm.remove_atm_grps([self])

    def __repr__(self):
        return '<AtomGroup: [%s]>' % ', '.join([str(x) for x in self.atoms])

    def __eq__(self, other):
        """Overrides the default implementation"""
        if type(self) == type(other):
            return (self.atoms == other.atoms
                    and self.features == other.features)
        return False

    def __ne__(self, other):
        """Overrides the default implementation"""
        return not self.__eq__(other)

    def __lt__(self, other):
        atms1 = tuple(sorted(self.atoms))
        atms2 = tuple(sorted(other.atoms))
        return atms1 < atms2

    def __len__(self):
        # Number of atoms.
        return self.size

    def __hash__(self):
        """Overrides the default implementation"""
        if self._hash_cache is None:
            # Transform atoms and features list into an imutable data
            # structure. The lists are sorted in order to avoid
            # dependence on appending order.
            atoms_tuple = tuple(self.atoms)
            feat_tuple = tuple(self.features)
            self._hash_cache = hash((atoms_tuple, feat_tuple, self.__class__))
        return self._hash_cache
