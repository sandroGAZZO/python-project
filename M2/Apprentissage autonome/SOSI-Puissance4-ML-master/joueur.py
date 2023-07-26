#!/usr/bin/python
from puissance4 import *
from minimax import *
import random
from keras.models import load_model
import operator



class joueur(object):

    def __init__(self,pion):
        self._pions = pion

    def choisirColonne(self,plateau):
        colonne = 0
        while (colonne < 1) or (colonne > plateau.LARGEUR):
            colonne = int(input("Vous êtes {}, quelle colonne ? \n".format(self._pions)))

        colonne = int(colonne)
        return colonne

class joueurIARandom(joueur):
    def choisirColonne(self,plateau):
        return random.randint(1, plateau.LARGEUR)

class joueurIAMinMax(joueur):
    def choisirColonne(self,plateau):
        board=plateau.plateau_en_board()
        coups = plateau.coupsPossibles()
        if len(coups)==1:
            return coups[0]

        sym=str(self._pions)
        col=minimax(board, 6, sym)[1]
        return col

class joueurIAReseauNeurone(joueur):
    def colonneFromKerasOutput(self, keras_output):
        index, value = max(enumerate(keras_output[0]), key=operator.itemgetter(1))
        return index

    def choisirColonne(self, plateau):
        coupsPossibles = plateau.coupsPossibles()
        model = load_model("data/keras_model_presentation")
        coups =  self.colonneFromKerasOutput(model.predict(plateau.plateauToKeras(), batch_size=1, verbose=0))
        while coups not in coupsPossibles:
            coups = random.randint(1, plateau.LARGEUR)

        return coups
