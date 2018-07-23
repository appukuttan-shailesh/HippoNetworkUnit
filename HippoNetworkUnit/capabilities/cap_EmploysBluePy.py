import sciunit
from sciunit import Capability

class EmploysBluePy(sciunit.Capability):
	"""Indicates that the model receives synapse"""

	def sample_bouton_density(self, appositions, num_samples, syn_per_bouton):
		""" should return a pandas dataframe, e.g.:
			                   1           2           3           4            5
			SP_AA     0.00140015  0.00021414   0.0122389   0.0334388    0.0400772
			SO_OLM      0.142811    0.185051    0.284134    0.220945     0.251706
			SP_CCKBC   0.0547306   0.0553962    0.386957    0.402159     0.410369
			SO_BS       0.209138    0.206002   0.0823694    0.211692      0.25851
		"""
		raise NotImplementedError()

	def get_bouton_density(self, appositions, num_samples, syn_per_bouton):
		df = self.sample_bouton_density(appositions, num_samples, syn_per_bouton)
		df['mean'] = df.mean(axis=1)
		df['std'] = df.std(axis=1)
		return df
