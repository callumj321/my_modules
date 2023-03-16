import MDAnalysis as mda
from MDAnalysis import analysis
from MDAnalysis.analysis import hbonds
from MDAnalysis.analysis import contacts
from MDAnalysis.analysis.contacts import contact_matrix
import pandas as pd
import numpy as np
import Bio.PDB
import mdtraj as md
import mdtraj.geometry as mdgeo

class traj_analysis:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,pdb_filename,xtc_filename,analyte='protein'):
        # Save these for later
        self.pdb_filename = pdb_filename
        self.xtc_filename = xtc_filename
        self.SS_types = ['H','B','E','G','I','T','S',' ']
        self.SS_types_simp = ['H','E','C']
        self.colour_list = [
            '#ebac23','#b80058','#008cf9','#006e00',
            '#00bbad','#d163e6','#b24502','#ff9287',
            '#5954d6','#00c6f8','#878500','#00a76c',
            '#bdbdbd','#000000','#b80058','#ebac23',
            '#b80058','#008cf9','#006e00','#00bbad',
            '#d163e6','#b24502','#ff9287','#5954d6',
            '#00c6f8','#878500','#00a76c','#bdbdbd',
            '#b80058']        self.analyte_sel = analyte
        print('Loading in mda universe...')
        self.mda_universe = mda.Universe(pdb_filename,xtc_filename) # <--- Load in universe
        # self.mdtraj_universe = md.load(self.xtc_filename,top=self.pdb_filename) # <--- Load in universe
        print('Done.')
        self.WW_hydro_scale = {'CYS': -0.24, 'ASP': 1.23, 'SER': 0.13, 'GLN': 	0.58, 
                               'LYS': 0.99, 'ILE': -0.31, 'PRO': 0.45, 'THR': 0.14, 
                               'PHE': -1.13, 'ASN': 0.42, 'GLY': 0.01, 'HSD': 0.96, 
                               'LEU': -0.56, 'ARG': 0.81, 'TRP': -1.85, 'ALA': 0.17, 
                               'VAL': 0.07, 'GLU': 2.02, 'TYR': -0.94, 'MET': -0.23}
        self.charge_scale = {'CYS': 0, 'ASP': -1, 'SER': 0, 'GLN': 0, 
                               'LYS': 1, 'ILE': 0, 'PRO': 0, 'THR': 0, 
                               'PHE': 0, 'ASN': 0, 'GLY': 0, 'HSD': 1, 
                               'LEU': 0, 'ARG': 1, 'TRP': 0, 'ALA': 0, 
                               'VAL': 0, 'GLU': -1, 'TYR': 0, 'MET': 0}
        self.AA_atoms = {'CYS': 10, 'ASP': 12, 'SER': 11, 'GLN': 19, 
                         'LYS': 22, 'ILE': 19, 'PRO': 14, 'THR': 14, 
                         'PHE': 20, 'ASN': 13, 'GLY': 7, 'HSD': 17, 
                         'LEU': 19, 'ARG': 24, 'TRP': 24, 'ALA': 10, 
                         'VAL': 16, 'GLU': 15, 'TYR': 21, 'MET': 17}
        self.C_AA_atoms = {'CYS': 3, 'ASP': 4, 'SER': 3, 'GLN': 5, 
                         'LYS': 6, 'ILE': 6, 'PRO': 5, 'THR': 4, 
                         'PHE': 9, 'ASN': 4, 'GLY': 2, 'HSD': 6, 
                         'LEU': 6, 'ARG': 6, 'TRP': 11, 'ALA': 3, 
                         'VAL': 5, 'GLU': 5, 'TYR': 9, 'MET': 5}
        self.CH_AA_atoms = {'CYS': 7, 'ASP': 8, 'SER': 8, 'GLN': 15, 
                         'LYS': 19, 'ILE': 17, 'PRO': 12, 'THR': 11, 
                         'PHE': 18, 'ASN': 9, 'GLY': 5, 'HSD': 13, 
                         'LEU': 17, 'ARG': 21, 'TRP': 22, 'ALA': 8, 
                         'VAL': 14, 'GLU': 11, 'TYR': 19, 'MET': 14}
    def load_mdtraj_uni(self):
        self.mdtraj_universe = md.load(self.xtc_filename, top='protein and segid A'.format(self.pdb_filename, segid))
    def help(self):
        print('SS types...')
        print(self.SS_types)
        print('Or if simplified...')
        print(self.SS_types_simp)
    def segid_to_resnames(self):
        resnames = self.mda_universe.select_atoms(self.analyte_sel).residues.resnames
        return resnames
    def mdtraj_pro_analysis_setup(self):
        pro = self.mda_universe.select_atoms(self.analyte_sel)
        mdtraj_u = md.load(self.xtc_filename,top=self.pdb_filename, atom_indices=pro.indices)
        protein = mdtraj_u.atom_slice(mdtraj_u.top.select('protein'))
        return protein
    def dssp(self,simplified=False):
        protein = self.mdtraj_pro_analysis_setup()
        ss_data = md.compute_dssp(protein, simplified=simplified) # <--- Do SS analysis
        print('Done DSSP')
        return ss_data # <--- Return analysis
    def sasa(self):
        protein = self.mdtraj_pro_analysis_setup()
        sasa_data = md.shrake_rupley(protein, mode='residue') # <--- Do SASA analysis
        print('Done DSSP')
        return sasa_data # <--- Return analysis
    def get_phi(self):
        protein = self.mdtraj_pro_analysis_setup()
        phi_indices = md.compute_phi(protein)
        print('Done angles')
        return phi_indices # <--- Return the angles
    def get_psi(self):
        protein = self.mdtraj_pro_analysis_setup()
        psi_indices = md.compute_psi(protein)
        print('Done angles')
        return psi_indices # <--- Return the angles
    def get_omega(self):
        protein = self.mdtraj_pro_analysis_setup()
        omega_indices = md.compute_omega(protein)
        print('Done angles')
        return omega_indices # <--- Return the angles
    def pro_hbond_analysis(self):
        # Select donor and acceptor atoms
        donor_atoms = self.mda_universe.select_atoms('protein and (name NH1 or name NH2)')
        acceptor_atoms = self.mda_universe.select_atoms('protein and (name O or name OXT)')
        # Compute hydrogen bonds
        hbonds = mda.analysis.hbonds.HydrogenBondAnalysis(self.mda_universe, 'protein', 'protein', distance=3.0, angle=120.0, pbc=True,donors=donor_atoms, acceptors=acceptor_atoms, exclude_water=True)
        hbonds.run()
        self.hbond_counts = hbonds.count_by_time()
        return self.hbond_counts
    def measure_RDG(self):
        out_file = self.xtc_filename.split('.')[0]+'_RDG.csv'
        protein = self.mda_universe.select_atoms('protein')
        n_frames = len(self.mda_universe.trajectory)
        n_atoms = len(protein)
        rg_array = np.zeros(n_frames)
        rg_array=[]
        for ts in self.mda_universe.trajectory:
            # Get the positions of the protein atoms in this frame
            positions = protein.positions
            
            # Calculate the center of mass of the protein
            com = np.mean(positions, axis=0)
            
            # Subtract the center of mass from the positions
            positions -= com
            
            # Calculate the squared distance of each atom from the center of mass
            distances = np.sum(positions**2, axis=1)
            
            # Calculate the radius of gyration using the squared distances
            rg = np.sqrt(np.sum(distances) / n_atoms)
            
            # Store the radius of gyration for this frame
            rg_array.append(rg)
        self.RDG = pd.DataFrame({'time':self.mdtraj_universe.time,'RDG':rg_array})
        self.RDG.to_csv(out_file,index=False)
    def angle(self,vector1,vector2):
        theta = np.rad2deg(np.arccos(np.dot(vector1, vector2)/(norm(vector1)*norm(vector2))))
        return theta
    def measure_angle(self,sel1,sel2,sel3,name_mark=''):
        out_file = self.xtc_filename.split('.')[0]+name_mark+'_angle.csv'
        angle_data = []
        # ---Make selections
        sel1_made = self.mda_universe.select_atoms(sel1)
        sel2_made = self.mda_universe.select_atoms(sel2)
        sel3_made = self.mda_universe.select_atoms(sel3)
        times = []
        for ts in self.mda_universe.trajectory:
            times.append(self.mda_universe.trajectory.time)
            # Get vectors for calculating angle
            vec1 = sel1_made.center_of_geometry() - sel2_made.center_of_geometry()
            vec2 = sel3_made.center_of_geometry() - sel2_made.center_of_geometry()
            # Get the angle
            degree = self.angle(vec1,vec2)
            # Add angle to data
            angle_data.append(degree)
        self.angle_data = pd.DataFrame({'time':times,'angle':angle_data})
        self.angle_data.to_csv(out_file,index=False)
        return self.angle_data