
import numpy as np
from scipy.stats import norm

class Tokenizer:
    def __init__(self, config):
        self.BIN_SIZE = config.bin_size
        self.MAX_COVERAGE = config.max_coverage
        self.bins, self.bin_values = self.get_gauusian_bins(self.BIN_SIZE, self.MAX_COVERAGE)
        self.vocab_size = len(self.bin_values)

    @staticmethod
    def get_gauusian_bins(bin_size, max_coverage):
        # returns bin boundaries and bin centers s.t. each bin contains
        # BIN_SIZE % of the total gauusian distribution.
        N = norm(loc=0, scale=1)
        def _get_next_bin_boundary(init_pt, bin_size):
            cdf_right_boundary = np.clip(N.cdf(init_pt) + bin_size, 0, 0.99999)
            return N.ppf(cdf_right_boundary)

        pos_bins = [0]; coverage = 0
        while coverage < max_coverage / 2:
            nxt_bin_boundary = _get_next_bin_boundary(
                init_pt=pos_bins[-1],
                bin_size=bin_size)
            pos_bins.append(nxt_bin_boundary)
            coverage = N.cdf(nxt_bin_boundary) - N.cdf(0)

        all_bins = np.array([-x for x in pos_bins[1::][::-1]] + pos_bins)
        bin_center = 0.5 * (all_bins[:-1] + all_bins[1:])

        return all_bins, bin_center

    @   staticmethod
    def norm_std(x, loc, scale):
        return ((x.T - loc) / (scale + 1e-6)).T
    
    @   staticmethod
    def denorm_std(x, loc, scale):
        return (x.T * (scale) + loc).T

    def clip(self, x):
        # clips all values in x  thath is smaller than the smallest 
        # bin or larger than the largest bin.
        return np.clip(x, self.bins[0]+(1e-3), self.bins[-1]-(1e-3))
    
    def encode(self, x, params=None, return_pt=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if params == None:    
            if x.ndim == 1:
                params = {'loc':x.mean(), 'scale':x.std()}
            elif x.ndim == 2:
                params = {'loc':x.mean(axis=1), 'scale':x.std(axis=1)}
            else:
                raise ValueError('Input must be 1d or 2d array')
        
        x = self.norm_std(x, **params)
        x = self.clip(x)
        token_ids = np.clip(np.digitize(x, self.bins, right=False)-1, 0, len(self.bins)-2)
        
        return token_ids, params
    
    def decode(self, tkn_id, params):
        # tkn_id is 2d mat of shape Batch x Sequence
        # params is a dict of whose values are 2d arrays of shape Batch x 1
        
        if not isinstance(tkn_id, np.ndarray):
            tkn_id = np.array(tkn_id)
        
        shape = tkn_id.shape
        values = self.bin_values[tkn_id.ravel()].reshape(shape)
        values = self.denorm_std(values, **params)
        return values
    
    def digitize(self, X):
        # wrapper function that does required clippings and digitizations 
        # of approriately standadized matrices.
        X_clipped = self.clip(X)
        return np.clip(np.digitize(X_clipped, self.bins, right=False)-1, 0, len(self.bins)-2)

    
