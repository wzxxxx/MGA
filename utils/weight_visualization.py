import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor, MolSurf
from rdkit.Chem.Draw import rdMolDraw2D, MolToFile, _moltoimg
import matplotlib.cm as cm
import matplotlib
from IPython.display import SVG, display
import seaborn as sns

sns.set(color_codes=True)


def weight_visulize_origin(smiles, atom_weight):
    print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('Oranges')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {}
    atom_raddi = {}
    # weight_norm = np.array(ind_weight).flatten()
    # threshold = weight_norm[np.argsort(weight_norm)[1]]
    # weight_norm = np.where(weight_norm < threshold, 0, weight_norm)
    
    for i in range(mol.GetNumAtoms()):
        if atom_weight[i] <= 0.3:
            atom_colors[i] = plt_colors.to_rgba(float(0.05))
        elif atom_weight[i] <= 0.4:
            atom_colors[i] = plt_colors.to_rgba(float(0.25))
        elif atom_weight[i] <= 0.5:
            atom_colors[i] = plt_colors.to_rgba(float(0.6))
        else:
            atom_colors[i] = plt_colors.to_rgba(float(1.0))
    rdDepictor.Compute2DCoords(mol)
    
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(1)
    op = drawer.drawOptions()
    
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, highlightAtoms=range(0, mol.GetNumAtoms()), highlightBonds=[],
                        highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg2 = svg.replace('svg:', '')
    svg3 = SVG(svg2)
    display(svg3)


def weight_visulize(smiles, atom_weight):
    print(smiles)
    atom_weight_list = atom_weight.squeeze().numpy().tolist()
    max_atom_weight_index = atom_weight_list.index(max(atom_weight_list))
    significant_weight = atom_weight[max_atom_weight_index]
    mol = Chem.MolFromSmiles(smiles)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('Oranges')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {}
    bond_colors = {}
    # weight_norm = np.array(ind_weight).flatten()
    # threshold = weight_norm[np.argsort(weight_norm)[1]]
    # weight_norm = np.where(weight_norm < threshold, 0, weight_norm)
    atom_new_weight = [0 for x in range(mol.GetNumAtoms())]
    # generate most important significant circle fragment and attach significant weight
    atom = mol.GetAtomWithIdx(max_atom_weight_index)
    # # find neighbors 1
    atom_neighbors_1 = [x.GetIdx() for x in atom.GetNeighbors()]
    # find neighbors 2
    atom_neighbors_2 = []
    for neighbors_1_index in atom_neighbors_1:
        neighbor_1_atom = mol.GetAtomWithIdx(neighbors_1_index)
        atom_neighbors_2 = atom_neighbors_2 + [x.GetIdx() for x in neighbor_1_atom.GetNeighbors()]
    atom_neighbors_2.remove(max_atom_weight_index)
    # find neighbors 3
    atom_neighbors_3 = []
    for neighbors_2_index in atom_neighbors_2:
        neighbor_2_atom = mol.GetAtomWithIdx(neighbors_2_index)
        atom_neighbors_3 = atom_neighbors_3 + [x.GetIdx() for x in neighbor_2_atom.GetNeighbors()]
    atom_neighbors_3 = [x for x in atom_neighbors_3 if x not in atom_neighbors_1]
    # attach neighbor 3 significant weight
    for i in atom_neighbors_3:
        atom_new_weight[i] = significant_weight*0.5
    for i in atom_neighbors_2:
        atom_new_weight[i] = significant_weight
    for i in atom_neighbors_1:
        atom_new_weight[i] = significant_weight
    atom_new_weight[max_atom_weight_index] = significant_weight

    significant_fg_index = [max_atom_weight_index] + atom_neighbors_1 + atom_neighbors_2 + atom_neighbors_3

    for i in range(mol.GetNumAtoms()):
        atom_colors[i] = plt_colors.to_rgba(float(atom_new_weight[i]))

    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        x = atom_new_weight[u]
        y = atom_new_weight[v]
        bond_weight = (x+y)/2
        if u in significant_fg_index and v in significant_fg_index:
            bond_colors[i] = plt_colors.to_rgba(float(abs(bond_weight)))
        else:
            bond_colors[i] = plt_colors.to_rgba(float(abs(0)))
    rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(1)
    op = drawer.drawOptions()

    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, highlightAtoms=range(0, mol.GetNumAtoms()), highlightBonds=range(0, mol.GetNumBonds()),
                        highlightAtomColors=atom_colors, highlightBondColors=bond_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg2 = svg.replace('svg:', '')
    svg3 = SVG(svg2)
    display(svg3)


def weight_visulize_py(smiles, atom_weight,):
    print(smiles)
    atom_weight_list = atom_weight.squeeze().numpy().tolist()
    max_atom_weight_index = atom_weight_list.index(max(atom_weight_list))
    significant_weight = atom_weight[max_atom_weight_index]
    mol = Chem.MolFromSmiles(smiles)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('Oranges')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {}
    bond_colors = {}
    # weight_norm = np.array(ind_weight).flatten()
    # threshold = weight_norm[np.argsort(weight_norm)[1]]
    # weight_norm = np.where(weight_norm < threshold, 0, weight_norm)
    atom_new_weight = [0 for x in range(mol.GetNumAtoms())]
    # generate most important significant circle fragment and attach significant weight
    atom = mol.GetAtomWithIdx(max_atom_weight_index)
    # # find neighbors 1
    atom_neighbors_1 = [x.GetIdx() for x in atom.GetNeighbors()]
    # find neighbors 2
    atom_neighbors_2 = []
    for neighbors_1_index in atom_neighbors_1:
        neighbor_1_atom = mol.GetAtomWithIdx(neighbors_1_index)
        atom_neighbors_2 = atom_neighbors_2 + [x.GetIdx() for x in neighbor_1_atom.GetNeighbors()]
    atom_neighbors_2.remove(max_atom_weight_index)
    # find neighbors 3
    atom_neighbors_3 = []
    for neighbors_2_index in atom_neighbors_2:
        neighbor_2_atom = mol.GetAtomWithIdx(neighbors_2_index)
        atom_neighbors_3 = atom_neighbors_3 + [x.GetIdx() for x in neighbor_2_atom.GetNeighbors()]
    atom_neighbors_3 = [x for x in atom_neighbors_3 if x not in atom_neighbors_1]
    # attach neighbor 3 significant weight
    for i in atom_neighbors_3:
        atom_new_weight[i] = significant_weight*0.5
    for i in atom_neighbors_2:
        atom_new_weight[i] = significant_weight
    for i in atom_neighbors_1:
        atom_new_weight[i] = significant_weight
    atom_new_weight[max_atom_weight_index] = significant_weight

    significant_fg_index = [max_atom_weight_index] + atom_neighbors_1 + atom_neighbors_2 + atom_neighbors_3

    for i in range(mol.GetNumAtoms()):
        atom_colors[i] = plt_colors.to_rgba(float(atom_new_weight[i]))

    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        x = atom_new_weight[u]
        y = atom_new_weight[v]
        bond_weight = (x+y)/2
        if u in significant_fg_index and v in significant_fg_index:
            bond_colors[i] = plt_colors.to_rgba(float(abs(bond_weight)))
        else:
            bond_colors[i] = plt_colors.to_rgba(float(abs(0)))
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(1)
    op = drawer.drawOptions()
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    smiles_name = eval(repr(smiles).replace('\\', '|'))
    MolToFile(mol, r'./CYP2D6/'+smiles_name+'.png', highlightAtoms=range(0, mol.GetNumAtoms()), highlightBonds=range(0, mol.GetNumBonds()),
              highlightAtomColors=atom_colors, highlightBondColors=bond_colors)

