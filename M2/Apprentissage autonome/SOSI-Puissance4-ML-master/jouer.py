import puissance4
from joueur import *
import afficheur


class partie(object):
    def __init__(self, joueur1, joueur2):
        self.joueur1 = joueur1
        self.joueur2 = joueur2
        self.joueur_actuel = self.joueur1
        self.plateau = puissance4.plateau()
        self.afficheur = afficheur.afficheurConsole(self.plateau)

    def lancerPartie(self):
        self.afficheur.afficherPlateau()
        while not(self.plateau.estFini()):
            self.plateau.lacherPion(self.joueur_actuel.choisirColonne(self.plateau), self.joueur_actuel._pions)
            self.afficheur.afficherPlateau()
            self.changerJoueur()

        self.changerJoueur()
        print("{} a gagné !".format(self.joueur_actuel._pions))

    def changerJoueur(self):
        if self.joueur_actuel is self.joueur1:
            self.joueur_actuel = self.joueur2
        else:
            self.joueur_actuel = self.joueur1

p = partie(joueurIAReseauNeurone(puissance4.pion.JAUNE),joueur(puissance4.pion.ROUGE))
p.lancerPartie()
