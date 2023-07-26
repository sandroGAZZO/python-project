import puissance4
from puissance4 import pion
import os

class afficheurConsole(object):
    def __init__(self, plateau):
        self.plateau = plateau

    def afficherPlateau(self):
        os.system('cls' if os.name=='nt' else 'clear')
        print("\n")
        for ligne in self.plateau.cases:
            for pion in ligne:
                print(pion, end="")
                # self.printPion(pion)
            print("\n")
