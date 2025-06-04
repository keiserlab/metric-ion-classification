import pymol

def highlight_ions_save_pse(pdb_file, ions, pse_file):
    # Initialize PyMOL
    pymol.finish_launching(['pymol', '-q'])

    # Load the PDB file
    pymol.cmd.load(pdb_file)

    # Iterate over the ions and highlight them
    for ion in ions:
        pymol.cmd.select('ion', 'elem ' + ion)
        pymol.cmd.show('spheres', 'ion')
        pymol.cmd.color('red', 'ion')

    # Show associated text for the highlighted ions
    pymol.cmd.set('label_position', '[x,y,z]')  # Replace [x,y,z] with desired label position
    pymol.cmd.set('label_color', 'white')
    pymol.cmd.set('label_size', -0.5)
    pymol.cmd.label('ion', '"ID: " + resi + ", Name: " + name')

    # Save the PyMOL session as a .pse file
    pymol.cmd.save(pse_file)

    # Quit PyMOL
    pymol.cmd.quit()

# Example usage
"""
pdb_file = 'path/to/your/pdb/file.pdb'
ions = ['Na', 'Cl']  # List of ions to highlight
pse_file = 'path/to/save/your/pse/file.pse'
highlight_ions_save_pse(pdb_file, ions, pse_file)
"""



