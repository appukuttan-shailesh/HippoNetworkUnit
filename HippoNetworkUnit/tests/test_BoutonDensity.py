import sciunit
import sciunit.scores
import HippoNetworkUnit.capabilities as cap
import HippoNetworkUnit.scores

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn
from datetime import datetime
import quantities
import os

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # Python 2 backport

#===============================================================================

class BoutonDensityTest(sciunit.Test):
    """Tests the bouton density for various morphology types"""
    score_type = score_type = HippoNetworkUnit.scores.CombineZScores

    def __init__(self,
                 observation="",
                 name="Bouton Density Test",
                 appositions=False,
                 num_samples=10,
                 syn_per_bouton=1.2):
        description = ("Tests the bouton density for various morphology types")
        required_capabilities = (cap.EmploysBluePy,)

        observation = pd.read_csv(observation, names=['mtype', 'bio_mean', 'bio_std'], skiprows=2, usecols=[0,1,2], delim_whitespace=True)
        self.figures = []
        sciunit.Test.__init__(self, observation, name)
        self.appositions = appositions
        self.num_samples = num_samples
        self.syn_per_bouton = syn_per_bouton
        self.directory_output = './BoutonDensityTest/'
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        print("Test Config:")
        print("\t Name: {}".format(self.name))
        print("\t appositions: {}".format(self.appositions))
        print("\t num_samples: {}".format(self.num_samples))
        print("\t syn_per_bouton: {}".format(self.syn_per_bouton))

    #----------------------------------------------------------------------

    def validate_observation(self, observation):
        pass

    #----------------------------------------------------------------------

    def generate_prediction(self, model, verbose=False):
        """Implementation of sciunit.Test.generate_prediction."""
        self.model_name = model.name
        prediction = model.get_bouton_density(self.appositions, self.num_samples, self.syn_per_bouton)
        return prediction

    #----------------------------------------------------------------------

    def compute_score(self, observation, prediction, verbose=False):
        """Implementation of sciunit.Test.score_prediction."""

        # compare match and evaluate score
        zscores = {}
        selected = observation['mtype'].values
        obs_pred_stats = observation.copy()
        obs_pred_stats['model_mean'] = prediction['mean'][selected].values
        obs_pred_stats['model_std'] = prediction['std'][selected].values
        obs_pred_stats.set_index('mtype', inplace=True)
        obs_pred_stats = obs_pred_stats.rename_axis(None,0).rename_axis('mtype',1)
        for mtype in list(observation.index):
            # currently observation has std = 0, while prediction has variable std
            # so evaluating Z-score below by interchanging positions.
            # Might need to re-examine and find a better statistic in future.
            zscores[mtype] = sciunit.scores.ZScore.compute({'mean':obs_pred_stats['model_mean'][mtype],
                                                            'std' :obs_pred_stats['model_std'][mtype]},
                                                            {'value' :obs_pred_stats['bio_mean'][mtype]}).score
        self.score = HippoNetworkUnit.scores.CombineZScores.compute(zscores.values())
        self.score.description = "Mean of absolute Z-scores"

        # create output directory
        self.path_test_output = os.path.join(self.directory_output, self.model_name, self.timestamp)
        Path(self.path_test_output).mkdir(parents=True, exist_ok=True)

        self.observation = observation
        self.prediction = prediction

        # create relevant output files
        # 1. Prediction Plot
        plt.close('all')
        fig, ax = plt.subplots()
        labels = list(prediction.index)
        ind = np.arange(len(labels))
        width = 0.75
        s = ax.bar(ind, prediction['mean'], width, yerr=prediction['std'])
        ax.set_xlabel('mtype')
        ax.set_ylabel('density (um^-1)')
        if self.appositions:
            title = 'apposition density'
        else:
            title = 'bouton density'
        ax.set_title(title)
        ax.set_xticks(ind)
        ax.set_xticklabels(labels, rotation='vertical')
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_test_output,title+'.pdf'))
        self.figures.append(os.path.join(self.path_test_output,title+'.pdf'))

        # 2. Prediction Table
        prediction.to_csv(os.path.join(self.path_test_output,"modelData.txt"), header=True, index=True, sep='\t')
        self.figures.append(os.path.join(self.path_test_output,"modelData.txt"))

        # 3. Stats Table
        obs_pred_stats.to_csv(os.path.join(self.path_test_output,"statsData.txt"), header=True, index=True, sep='\t')
        self.figures.append(os.path.join(self.path_test_output,"statsData.txt"))

        # 4. Error Plot
        plt.close('all')
        x = obs_pred_stats['model_mean'].values
        y = obs_pred_stats['bio_mean'].values
        l = np.linspace(0, max(x.max(), y.max()), 50)
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o')
        ax.errorbar(x, y, xerr=obs_pred_stats['model_std'].values, yerr=obs_pred_stats['bio_std'].values, fmt='o', ecolor='g', capthick=2)
        ax.plot(l, l, 'k--')
        ax.set_xlabel('Model (um^-1)')
        ax.set_ylabel('Experiment (um^-1)')
        if self.appositions:
            title = 'apposition density'
        else:
            title = 'bouton density'
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_test_output,title+'_validation.pdf'))
        self.figures.append(os.path.join(self.path_test_output,title+'_validation.pdf'))

        return self.score

    #----------------------------------------------------------------------

    def bind_score(self, score, model, observation, prediction):
        score.related_data["figures"] = self.figures
        return score
