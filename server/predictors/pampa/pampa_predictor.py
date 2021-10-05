import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from rdkit import Chem
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '../chemprop')
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.train import predict
from rdkit.Chem import PandasTools
import random
import string
from rdkit.Chem.rdchem import Mol
from numpy import array
from typing import Tuple
from ..features.morgan_fp import MorganFPGenerator
from ..utilities.utilities import get_processed_smi, get_interpretation
from . import pampa_gcnn_scaler, pampa_gcnn_model
from ..base.gcnn import GcnnBase
import time


class PAMPAPredictior(GcnnBase):
    """
    Makes PAMPA permeability preditions

    Attributes:
        df (DataFrame): DataFrame containing column with smiles
        smiles_column_index (int): index of column containing smiles
        predictions_df (DataFrame): DataFrame hosting all predictions
    """

    def __init__(self, kekule_smiles: array = None, smiles: array = None):
        """
        Constructor for PAMPAPredictior class

        Parameters:
            kekule_smiles (Array): numpy array of RDkit molecules
        """

        GcnnBase.__init__(self, kekule_smiles, column_dict_key='Predicted Class (Probability)', columns_dict_order=1, smiles=smiles)

        self._columns_dict['Prediction'] = {
            'order': 2,
            'description': 'class label',
            'isSmilesColumn': False
        }

        self._columns_dict['Glowing Molecule'] = {
            'order': 4,
            'description': 'glowing molecule',
            'isSmilesColumn': True
        }

        self.model_name = 'pampa'

    def get_predictions(self) -> DataFrame:
        """
        Function that calculates consensus predictions

        Returns:
            Predictions (DataFrame): DataFrame with all predictions
        """

        if len(self.kekule_smiles) > 0:

            start = time.time()
            gcnn_predictions, gcnn_labels = self.gcnn_predict(pampa_gcnn_model, pampa_gcnn_scaler)
            end = time.time()
            print(f'{end - start} seconds to PAMPA predict {len(self.predictions_df.index)} molecules')

            self.predictions_df['Prediction'] = pd.Series(
                pd.Series(np.where(gcnn_predictions>=0.5, 'low or moderate permeability', 'high permeability'))
            )

            # trying to fit interpretation here
            kekule_smiles_df = pd.DataFrame(self.kekule_smiles, columns =['smiles'])
            intrprt_df = get_interpretation(kekule_smiles_df)
            self.predictions_df['Glowing Molecule'] = intrprt_df[['smiles', 'rationale_smiles']].agg('_'.join, axis=1)


        return self.predictions_df
