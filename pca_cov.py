'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
Ruby Nunez
May 27 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset.
        
        '''

        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

    def get_prop_var(self):
        return self.prop_var

    def get_cum_var(self):
        return self.cum_var

    def get_eigenvalues(self):
        return self.e_vals

    def get_eigenvectors(self):
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`
        '''

        centered_data = data - np.mean(data, axis=0)
        cov_matrix = (centered_data.T @ centered_data) / (data.shape[0] - 1)

        return cov_matrix


    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''

        total_var = sum(e_vals)
        prop_var = [(eigval / total_var) for eigval in e_vals]

        return prop_var


    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''

        accum_var = []
        accum_sum = 0
        for i in range(len(prop_var)):
            accum_sum += prop_var[i]
            accum_var.append(accum_sum)

        return accum_var


    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.
        '''

        self.vars = vars
        selected_data = self.data[vars].values

        if normalize:
            max_vals = np.max(selected_data, axis=0)
            min_vals = np.min(selected_data, axis=0)
            dynamic_range = max_vals - min_vals
            normalized_data = (selected_data - min_vals) / dynamic_range
            self.max_vals = max_vals
            self.min_vals = min_vals
            self.dynamic_range = dynamic_range
            self.normalized = True
            self.A = normalized_data
        else:
            self.normalized = False
            self.A = selected_data

        cov_matrix = self.covariance_matrix(self.A)
        e_vals, e_vecs = np.linalg.eig(cov_matrix)
        sorted_indices = np.argsort(e_vals)[::-1]
        sorted_e_vals = e_vals[sorted_indices]
        sorted_e_vecs = e_vecs[:, sorted_indices]

        self.e_vals = sorted_e_vals
        self.e_vecs = sorted_e_vecs

        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)


    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).
        '''

        if num_pcs_to_keep is None:
            num_pcs_to_keep = len(self.prop_var)

        x = range(1, num_pcs_to_keep + 1)
        y = self.cum_var[:num_pcs_to_keep]

        plt.plot(x, y, marker='o', markersize=8)
        plt.xlabel('Top Principal Components')
        plt.ylabel('Proportion Variance Accounted For')
        plt.title('Elbow Plot')


    def variance_accounted_for(self, desired_variance):
        '''Calculates the number of principal components needed to reach the desired variance accounted for.

        Parameters:
        -----------
        desired_variance: float
            The desired proportion variance accounted for (between 0 and 1).

        Returns:
        -----------
        int
            The number of principal components needed to reach the desired variance.
        '''

        for i, var in enumerate(self.cum_var):
            if var >= desired_variance:
                index = i
                break
        return index + 1


    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.
        '''

        selected_e_vecs = self.e_vecs[:, pcs_to_keep]
        pca_proj = self.A @ selected_e_vecs
        self.A_proj = pca_proj

        return pca_proj


    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)
        '''

        selected_e_vecs = self.e_vecs[:, :top_k]
        pca_proj = self.A @ selected_e_vecs
        data_proj = pca_proj @ selected_e_vecs.T

        if self.normalized == True:
            data_proj = (data_proj * self.dynamic_range) + self.min_vals

        return data_proj