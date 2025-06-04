from Bio.PDB.Selection import unfold_entities
from chimerax.atomic import selected_residues, selected_atoms
from chimerax.core.commands import run
from itertools import product


def select_proximal(session, center, radius, level='r', include_center=False, split_ionwaters = False):
    if include_center:
        run(session, f'select zone {center} {radius} extend true residues {str(level=="r").lower()}')
    else:
        # Exclude H only if we’re NOT selecting the center
        run(session, f'select zone {center} {radius} ~H extend false residues {str(level=="r").lower()}')
            
    # split into residue neighbors and ion/solvent neighbors
    if split_ionwaters:
        run(session, 'name frozen allproximal sel')

        ionsolvent = selected_atoms(run(session, 'select allproximal & (solvent | ions)'))
        proxres = selected_atoms(run(session, 'select allproximal & ~(solvent | ions)'))
        return set([a.residue for a in ionsolvent]), set([a.residue for a in proxres])

    if level == 'r':
        return selected_residues(session)


def get_contacts_with(source,
                      target=None,
                      entity=None,
                      radius=6.2,
                      level='A'):

    entity = entity or source.get_parent_by_level("M")
    
    source_atoms = unfold_entities([source], 'A')

    target_atoms = []
    if target is None:
        target_atoms = list(entity.get_atoms())
    else:
        if target.level == "A":
            target_atoms = [target]
        else:
            target_residues = unfold_entities(target, 'R')
            target_atoms = [a for r in target_residues
                                for a in r.get_unpacked_list()]

    ns = NeighborSearch(target_atoms)
    entities = set()
    for atom in source_atoms:
        entity = atom.get_parent_by_level(level)
        nb_entities = ns.search(atom.coord, radius, level)
        pairs = set(product([entity], nb_entities))
        entities.update(pairs)

    return entities


def get_proximal_compounds(source, radius=2.2):
    model = source.get_parent_by_level('M')
    proximal = get_contacts_with(source, entity=model,
                                 radius=radius, level='R')

    # Sorted by the compound order as in the PDB.
    return sorted(list(set([p[1] for p in proximal])),
                  key=lambda r: (r.parent.parent.id, r.parent.id, r.idx))