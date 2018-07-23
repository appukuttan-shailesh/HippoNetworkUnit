import sciunit
import os
import json
import numpy as np
import HippoNetworkUnit.capabilities as cap

class CA1_laminar_distribution_synapses(sciunit.Model):

    def __init__(self, name="CA1_laminar_distribution_synapses", CA1_laminar_distribution_synapses_info={}):

        sciunit.Model.__init__(self, name=name)
        self.name = name
        self.description = "HBP Hippocampus CA1's output to test synapses distribution across CA1 layers"
        self.CA1_laminar_distribution_synapses_info = CA1_laminar_distribution_synapses_info
        self.set_CA1_laminar_distribution_synapses_info_default()

    def set_CA1_laminar_distribution_synapses_info_default(self):
        model_prediction_path = "./models/model_predictions/CA1_laminar_distribution_synapses_HBPmod.json"
        with open(model_prediction_path, 'r') as fp:
            data = json.load(fp)
        self.CA1_laminar_distribution_synapses_info = data

    def get_CA1_laminar_distribution_synapses_info(self):
        return self.CA1_laminar_distribution_synapses_info


# ==============================================================================


class CA1Layers_NeuritePathDistance(sciunit.Model):

    def __init__(self, name='CA1Layers_NeuritePathDistance', CA1LayersNeuritePathDistance_info={}):
        self.CA1LayersNeuritePathDistance_info = CA1LayersNeuritePathDistance_info
        sciunit.Model.__init__(self, name=name)
        self.name = name
        self.description = "Dummy model to test neurite path-distances across CA1 layers"
        self.set_CA1LayersNeuritePathDistance_info_default()

    def set_CA1LayersNeuritePathDistance_info_default(self):
        self.CA1LayersNeuritePathDistance_info = {"SLM": {'PathDistance': {'value':'120 um'}},
                                                  "SR": {'PathDistance': {'value':'280 um'}},
                                                  "SP": {'PathDistance': {'value':'40 um'}},
                                                  "SO": {'PathDistance': {'value':'100 um'}}
                                                 }

    def get_CA1LayersNeuritePathDistance_info(self):
        return self.CA1LayersNeuritePathDistance_info

# ==============================================================================

class BluePy_Circuit_Loader(sciunit.Model, cap.EmploysBluePy):
    def __init__(self, name="BluePy_Circuit_Loader", description="Circuit level model that uses BluePy", root_path=None):
        try:
            import bluepy
        except ImportError:
            print("Please install the following package: bluepy")
            return
        try:
            import pandas
        except ImportError:
            print("Please install the following package: pandas")
            return

        sciunit.Model.__init__(self, name=name)
        self.description = description
        if root_path == None:
            raise ValueError("Please specify the root path to the circuit!")
        if not os.path.isdir(root_path):
            raise ValueError("Specified root path to the circuit is invalid!")
        self.root_path = root_path

    def sample_bouton_density(self, appositions=False, num_samples=10, syn_per_bouton=1.2):
        import pandas as pd
        import bluepy
        from bluepy.v2.enums import Cell

        if appositions:
            circuit_path = self.root_path + 'CircuitConfig_struct'
        else:
            circuit_path = self.root_path + 'CircuitConfig'

        circ = bluepy.Circuit(circuit_path)
        mtypes = circ.v2.cells.mtypes
        df = pd.DataFrame(index=mtypes, columns=np.arange(num_samples)+1)
        for mtype in mtypes:
            gids = circ.v2.cells.ids(group={Cell.MTYPE: mtype, '$target': 'mc2_Column'}, limit=num_samples)
            # gids = circ.v2.cells.ids(group={Cell.MTYPE: mtype, '$target': 'cylinder'}, limit=num_samples)
            df.loc[mtype][:len(gids)] = circ.v2.stats.sample_bouton_density(num_samples, group=gids, synapses_per_bouton=syn_per_bouton)
        return df
