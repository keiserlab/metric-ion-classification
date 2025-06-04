from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.kdtrees import KDTree

from .features import ChemicalFeature

import numpy as np
from itertools import product

class InteractionType:
    def __init__(self,
                 src_grp,
                 trgt_grp,
                 inter_type,
                 src_interacting_atms=None,
                 trgt_interacting_atms=None,
                 src_centroid=None,
                 trgt_centroid=None,
                 directional=False,
                 params=None):

        self._src_grp = src_grp
        self._trgt_grp = trgt_grp

        src_interacting_atms = src_interacting_atms or []
        self._src_interacting_atms = list(src_interacting_atms)

        trgt_interacting_atms = trgt_interacting_atms or []
        self._trgt_interacting_atms = list(trgt_interacting_atms)

        self._src_centroid = (np.array(src_centroid)
                              if src_centroid is not None else None)
        self._trgt_centroid = (np.array(trgt_centroid)
                               if trgt_centroid is not None else None)

        self._type = inter_type
        self.directional = directional
        self._params = params or {}
        self._hash_cache = None

        self._apply_refs()
        self._expand_dict()

    @property
    def src_grp(self):
        return self._src_grp

    @src_grp.setter
    def src_grp(self, atm_grp):
        self._src_grp = atm_grp
        self._hash_cache = None
        self._apply_refs()

    @property
    def trgt_grp(self):
        return self._trgt_grp

    @trgt_grp.setter
    def trgt_grp(self, atm_grp):
        self._trgt_grp = atm_grp

        # Reset hash.
        self._hash_cache = None
        self._apply_refs()

    @property
    def src_interacting_atms(self):
        return self._src_interacting_atms or self.src_grp.atoms

    @property
    def trgt_interacting_atms(self):
        return self._trgt_interacting_atms or self.trgt_grp.atoms

    @property
    def src_centroid(self):
        if self._src_centroid is None:
            if self._src_interacting_atms:
                self._src_centroid = \
                    centroid(atom_coordinates(self._src_interacting_atms))
            else:
                src_centroid = self._src_grp.centroid
                self._src_centroid = src_centroid
        return self._src_centroid

    @src_centroid.setter
    def src_centroid(self, centroid):
        if centroid is None:
            self._src_centroid = None
        else:
            self._src_centroid = np.array(centroid)

    @property
    def trgt_centroid(self):
        if self._trgt_centroid is None:
            if self._trgt_interacting_atms:
                self._trgt_centroid = \
                    centroid(atom_coordinates(self._trgt_interacting_atms))
            else:
                trgt_centroid = self._trgt_grp.centroid

                self._trgt_centroid = trgt_centroid
        return self._trgt_centroid

    @trgt_centroid.setter
    def trgt_centroid(self, centroid):
        if centroid is None:
            self._trgt_centroid = None
        else:
            self._trgt_centroid = np.array(centroid)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, new_type):
        self._type = new_type
        self._hash_cache = None

    @property
    def params(self):
        """dict: Interaction parameters (distances, angles, etc)."""
        return self._params

    def get_partner(self, comp):
        if comp == self.src_grp:
            return self.trgt_grp
        elif comp == self.trgt_grp:
            return self.src_grp
        return None

    def is_directional(self):
        return self.directional

    def is_intramol_interaction(self):
        comps1 = self.src_grp.compounds
        comps2 = self.trgt_grp.compounds
        return len(comps1) == 1 and len(comps2) == 1 and comps1 == comps2

    def is_intermol_interaction(self):
        return not self.is_intramol_interaction()

    def _apply_refs(self):
        self.src_grp.add_interactions([self])
        self.trgt_grp.add_interactions([self])

    def clear_refs(self):
        self.src_grp.remove_interactions([self])
        self.trgt_grp.remove_interactions([self])

    def _expand_dict(self):
        for key in self._params:
            self.__dict__[key] = self._params[key]

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            is_equal_compounds = ((self.src_grp == other.src_grp
                                   and self.trgt_grp == other.trgt_grp)
                                  or (self.src_grp == other.trgt_grp
                                      and self.trgt_grp == other.src_grp))

            is_equal_interactions = self.type == other.type
            has_equal_params = self.params == other.params

            return (is_equal_compounds and is_equal_interactions
                    and has_equal_params)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):

        if self._hash_cache is None:
            params_values = []
            for key in sorted(self.params):
                if type(self.params[key]) is list:
                    val = tuple(self.params[key])
                else:
                    val = self.params[key]
                params_values.append(val)
            params_as_tuple = tuple(params_values)
            comp_values_as_tuple = tuple(sorted([self.src_grp, self.trgt_grp]))
            self._hash_cache = hash(tuple([comp_values_as_tuple,
                                           self.type, params_as_tuple]))
        return self._hash_cache

    def __repr__(self):
        return ('<InteractionType: compounds=(%s, %s) type=%s>'
                % (self.src_grp, self.trgt_grp, self.type))

class ProximalInteractionConfig(dict):
    def __init__(self):
        config = {'max_dist_proximal':6,
                  'min_dist_proximal':2,
                  'bsite_cutoff':6.2}
        super().__init__(config)


class InteractionsManager:
    def __init__(self, interactions=None, entry=None):
        if interactions is None:
            interactions = []

        self.entry = entry
        self._interactions = list(interactions)

    @property
    def interactions(self):
        return self._interactions

    def get_all_atm_grps(self):
        atm_grps = set()
        for inter in self.interactions:
            atm_grps.add(inter.src_grp)
            atm_grps.add(inter.trgt_grp)
        return atm_grps

    def add_interactions(self, interactions):
        self._interactions = list(set(self.interactions + list(interactions)))

    def remove_interactions(self, interactions):
        self._interactions = list(set(self.interactions) - set(interactions))

        for inter in interactions:
            inter.clear_refs()

    def __len__(self):
        # Number of interactions
        return len(self._interactions)

    def __iter__(self):
        """Iterate over children."""
        for inter in self.interactions:
            yield inter

    def filter_by_types(self, types):
        for inter in self.interactions:
            if inter.type in types:
                yield inter

class InteractionFilter:

    def __init__(self, ignore_self_inter=True, ignore_intra_chain=True,
                 ignore_inter_chain=True, ignore_res_res=True,
                 ignore_res_nucl=True, ignore_res_hetatm=True,
                 ignore_nucl_nucl=True, ignore_nucl_hetatm=True,
                 ignore_hetatm_hetatm=True, ignore_h2o_h2o=True,
                 ignore_any_h2o=False, ignore_multi_comps=False,
                 ignore_mixed_class=False):

        self.ignore_self_inter = ignore_self_inter
        self.ignore_intra_chain = ignore_intra_chain
        self.ignore_inter_chain = ignore_inter_chain
        self.ignore_res_res = ignore_res_res
        self.ignore_res_nucl = ignore_res_nucl
        self.ignore_res_hetatm = ignore_res_hetatm
        self.ignore_nucl_nucl = ignore_nucl_nucl
        self.ignore_nucl_hetatm = ignore_nucl_hetatm
        self.ignore_hetatm_hetatm = ignore_hetatm_hetatm
        self.ignore_h2o_h2o = ignore_h2o_h2o
        self.ignore_any_h2o = ignore_any_h2o
        self.ignore_multi_comps = ignore_multi_comps
        self.ignore_mixed_class = ignore_mixed_class



    @classmethod
    def new_pli_filter(cls, ignore_res_hetatm=False,
                       ignore_hetatm_hetatm=False, ignore_any_h2o=False,
                       ignore_self_inter=False, **kwargs):
        return cls(ignore_res_hetatm=ignore_res_hetatm,
                   ignore_hetatm_hetatm=ignore_hetatm_hetatm,
                   ignore_any_h2o=ignore_any_h2o,
                   ignore_self_inter=ignore_self_inter, **kwargs)

    def is_valid_pair(self, src_grp, trgt_grp):
        if src_grp == trgt_grp:
            return False

        if src_grp.contain_group(trgt_grp) or trgt_grp.contain_group(src_grp):
            return False

        has_multi_comps = (len(src_grp.compounds) > 1
                           or len(trgt_grp.compounds) > 1)
        if self.ignore_multi_comps and has_multi_comps:
            return False

        has_any_mixed = (src_grp.is_mixed() or trgt_grp.is_mixed())
        if self.ignore_mixed_class and has_any_mixed:
            return False

        is_same_compounds = \
            len(src_grp.compounds.intersection(trgt_grp.compounds)) >= 1
        if self.ignore_self_inter and is_same_compounds:
            return False

        is_same_chain = (src_grp.get_chains() == trgt_grp.get_chains()
                         and len(src_grp.get_chains()) == 1)

        is_res_res = (src_grp.is_residue() and trgt_grp.is_residue())
        if is_res_res:
            if self.ignore_res_res:
                return False
            elif self.ignore_intra_chain and is_same_chain:
                return False
            elif self.ignore_inter_chain and not is_same_chain:
                return False

        is_res_nucl = ((src_grp.is_residue() and trgt_grp.is_nucleotide())
                       or (src_grp.is_nucleotide() and trgt_grp.is_residue()))
        if self.ignore_res_nucl and is_res_nucl:
            return False

        is_res_hetatm = ((src_grp.is_residue() and trgt_grp.is_hetatm())
                         or (src_grp.is_hetatm() and trgt_grp.is_residue()))
        if self.ignore_res_hetatm and is_res_hetatm:
            return False

        is_nucl_nucl = (src_grp.is_nucleotide() and trgt_grp.is_nucleotide())
        if is_nucl_nucl:
            if self.ignore_nucl_nucl:
                return False
            elif self.ignore_intra_chain and is_same_chain:
                return False
            elif self.ignore_inter_chain and not is_same_chain:
                return False

        is_nucl_hetatm = ((src_grp.is_nucleotide() and trgt_grp.is_hetatm())
                          or (src_grp.is_hetatm()
                              and trgt_grp.is_nucleotide()))
        if self.ignore_nucl_hetatm and is_nucl_hetatm:
            return False

        is_hetatm_hetatm = (src_grp.is_hetatm() and trgt_grp.is_hetatm())
        if self.ignore_hetatm_hetatm and is_hetatm_hetatm:
            return False

        is_any_h2o = (src_grp.is_water() or trgt_grp.is_water())
        if self.ignore_any_h2o and is_any_h2o:
            return False

        is_h2o_h2o = (src_grp.is_water() and trgt_grp.is_water())
        if self.ignore_h2o_h2o and is_h2o_h2o:
            return False

        return True

    def is_valid_pair(self, src_grp, trgt_grp):
        if src_grp.contain_group(trgt_grp) or trgt_grp.contain_group(src_grp):
            return False

        is_same_compounds = \
            len(src_grp.compounds.intersection(trgt_grp.compounds)) >= 1
        if self.ignore_self_inter and is_same_compounds:
            return False

        return True

class InteractionCalculator:

    def __init__(self, inter_config=ProximalInteractionConfig(),
                inter_filter=None, 
                inter_funcs=None,
                add_h2o_pairs_with_no_target=False,
                add_dependent_inter=False,
                strict_donor_rules=True,
                strict_weak_donor_rules=True,
                lazy_comps_list=[]):

        self.inter_config = inter_config
        self.inter_filter = inter_filter
        self._inter_funcs = inter_funcs
        self.add_h2o_pairs_with_no_target = add_h2o_pairs_with_no_target
        self.add_non_cov = True
        self.add_dependent_inter = add_dependent_inter

        self.strict_donor_rules = strict_donor_rules
        self.strict_weak_donor_rules = strict_weak_donor_rules
        self.lazy_comps_list = lazy_comps_list or []

    @property
    def funcs(self):
        """dict: The dict that defines functions to calculate interactions."""
        return self._inter_funcs

    @funcs.setter
    def funcs(self, funcs):
        self._inter_funcs = funcs

    def calc_interactions(self, trgt_atm_grps, nb_atm_grps=None):
        nb_comp_grps = nb_atm_grps or trgt_atm_grps

        # Define the scope of the neighborhood search.
        ss = AtomGroupNeighborhood(nb_comp_grps, 10)

        computed_pairs = set()
        all_interactions = []

        bsite_cutoff = 6.2

        for trgt_atm_grp in trgt_atm_grps:
            for nb_atm_grp in ss.search(trgt_atm_grp.centroid, bsite_cutoff):
                if trgt_atm_grp == nb_atm_grp:
                    continue

                # If the pair has already been calculated.
                if ((trgt_atm_grp, nb_atm_grp) in computed_pairs
                        or (nb_atm_grp, trgt_atm_grp) in computed_pairs):
                    continue

                # If no filter was informed, it will accept everything.
                if (self.inter_filter is not None
                        and not self.inter_filter.is_valid_pair(trgt_atm_grp,
                                                                nb_atm_grp)):
                    continue

                computed_pairs.add((trgt_atm_grp, nb_atm_grp))

                feat_pairs = list(product(trgt_atm_grp.features,
                                          nb_atm_grp.features))
                feat_pairs = filter(lambda x: self.is_feature_pair_valid(*x),
                                    feat_pairs)

                is_intramol_inter = \
                    self._is_intramol_inter(trgt_atm_grp, nb_atm_grp)
                shortest_path_length = None

                for pair in feat_pairs:
                    if (pair[0].name != "Atom" and pair[1].name != "Atom"
                            and is_intramol_inter):

                        if shortest_path_length is None:
                            cutoff = self.inter_config.get("min_bond_separation", 0)
                            shortest_path_length = trgt_atm_grp.get_shortest_path_length(nb_atm_grp, cutoff)

                        if shortest_path_length != float('inf'):
                            continue

                    calc_inter_params = (trgt_atm_grp, nb_atm_grp) + pair
                    interactions = \
                        self._resolve_interactions(*calc_inter_params)
                    all_interactions.extend(interactions)

        if self.add_dependent_inter:
            dependent_interactions = \
                self.find_dependent_interactions(all_interactions)
            all_interactions.extend(dependent_interactions)

        # Get only unique interactions.
        all_interactions = set(all_interactions)

        #self.remove_inconsistencies(all_interactions)

        if not self.add_h2o_pairs_with_no_target:
            self.remove_h2o_pairs_with_no_target(all_interactions)

        return InteractionsManager(all_interactions)

    def _resolve_interactions(self, group1, group2, feat1, feat2):
        funcs = self.get_functions(feat1.name, feat2.name)
        if len(funcs) == 0:
            raise IllegalArgumentError("It does not exist a corresponding "
                                       "function to the features: '%s' and "
                                       "'%s'." % (feat1, feat2))

        interactions = []
        for func in funcs:
            result = func(self, (group1, group2, feat1, feat2)) or []
            interactions.extend(result)

        return interactions

    def _is_intramol_inter(self, grp1, grp2):
        comps1 = grp1.compounds
        comps2 = grp2.compounds
        return len(comps1) == 1 and len(comps2) == 1 and comps1 == comps2

    def is_within_boundary(self, value, key, func):
        if key not in self.inter_config:
            return True
        return func(value, self.inter_config[key])

    def is_feature_pair_valid(self, feat1, feat2):
        if isinstance(feat1, ChemicalFeature):
            feat1 = feat1.name
        if isinstance(feat2, ChemicalFeature):
            feat2 = feat2.name

        if not self.add_non_cov and (feat1 != "Atom" or feat2 != "Atom"):
            return False

        funcs = self.funcs
        return (True if ((feat1, feat2) in funcs
                         or (feat2, feat1) in funcs) else False)

    def get_functions(self, feat1, feat2):
        if isinstance(feat1, ChemicalFeature):
            feat1 = feat1.name
        if isinstance(feat2, ChemicalFeature):
            feat2 = feat2.name

        funcs = self.funcs
        if (feat1, feat2) in funcs:
            return funcs[(feat1, feat2)]
        elif (feat2, feat1) in funcs:
            return funcs[(feat2, feat1)]
        else:
            return None

    def set_functions_to_pair(self, pair, funcs):
        self.funcs[pair] = funcs

class AtomGroupNeighborhood:

    def __init__(self, atm_grps, bucket_size=10):
        self.atm_grps = list(atm_grps)


        # get the coordinates
        coord_list = [ga.centroid for ga in self.atm_grps]

        self.coords = np.array(coord_list, dtype="d")
        assert bucket_size > 1
        assert self.coords.shape[1] == 3
        self.kdt = KDTree(self.coords, bucket_size)

    def search(self, center, radius):
        """Return all atom groups in ``atm_grps`` that is up to a maximum of
        ``radius`` away (measured in Å) of ``center``.

        For atom groups with more than one atom, their centroid is used as a
        reference.
        """

        points = self.kdt.search(center.astype("d"), radius)
        
        n_grps_list = [self.atm_grps[point.index] for point in points]
        return n_grps_list
