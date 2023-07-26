from enum import Enum
import numpy as np
class pion(Enum):
    """Enumération représentant les pions que l'on place dans les colonnes du puissance 4"""
    VIDE = 0
    JAUNE = 1
    ROUGE = 2

    def __str__(self):
        return {
            pion.VIDE : " _ ",
            pion.JAUNE : " O ",
            pion.ROUGE : " X "
        }.get(self)



class plateau(object):
    LARGEUR = 7
    HAUTEUR = 6

    def __init__(self):
        self.cases = [[pion.VIDE]*self.LARGEUR for elem in range(self.HAUTEUR)]

    def coupsPossibles(self):
        col = []
        for i in range(1,8):
            if self.estVide(1,i):
                col.append(i)
        return col

    def lacherPion(self, colonne, pion):
        ligne = self.HAUTEUR
        while not(self.estVide(ligne, colonne)) and ligne > 0:
            ligne -= 1

        if ligne>0:
            self.placerPion(ligne, colonne, pion)
            return True;
        else:
            print("Colonne pleine")
            return False;

    def estVide(self, ligne, colonne):
        return self.obtenirPion(ligne, colonne) == pion.VIDE

    def obtenirPion(self, ligne, colonne):
        """Donne un pion à partir d'une ligne (première ligne en haut à gauche) et d'une colonne (première colonne à gauche)"""
        return self.cases[ligne-1][colonne-1]

    def placerPion(self, ligne, colonne, pion):
        self.cases[ligne-1][colonne-1] = pion

    def estFini(self):
        alignement = False;
        for i in range(1, self.HAUTEUR+1):
            for j in range(1, self.LARGEUR+1):
                if self.obtenirPion(i,j) != pion.VIDE:
                    if (self.alignementVertical(i,j) or self.alignementHorizontal(i,j) or self.alignementDiagonal(i,j)):
                        alignement = True;
                        return alignement
        return alignement

    def alignementVertical(self, ligne, colonne):
        #print("checking vert")
        QuatreEnColonne = False
        pionsConsecutifs = 0

        for i in range(ligne, self.HAUTEUR+1):
            if (self.obtenirPion(i, colonne) == self.obtenirPion(ligne, colonne)) and not(self.estVide(ligne,colonne)):
                pionsConsecutifs += 1
            else:
                break

        if pionsConsecutifs >= 4:
            QuatreEnColonne = True

        return QuatreEnColonne

    def alignementHorizontal(self, ligne, colonne):
        #print("checking vert")
        QuatreEnLigne = False
        pionsConsecutifs = 0

        for i in range(colonne, self.LARGEUR+1):
            if (self.obtenirPion(ligne, i) == self.obtenirPion(ligne, colonne)) and not(self.estVide(ligne,colonne)):
                pionsConsecutifs += 1
            else:
                break

        if pionsConsecutifs >= 4:
            QuatreEnLigne = True

        return QuatreEnLigne

    def alignementDiagonal(self, ligne, colonne):
        QuatreEnDiagonale = False
        count = 0

        # check for diagonals with positive slope
        pionsConsecutifs = 0
        j = colonne
        for i in range(ligne, 0, -1):
            if j > self.LARGEUR:
                break
            elif (self.obtenirPion(i,j) == self.obtenirPion(ligne, colonne)) and not(self.estVide(ligne,colonne)):
                pionsConsecutifs += 1
            else:
                break
            j += 1 # increment colonneumn when ligne is incremented

        if pionsConsecutifs >= 4:
            count += 1

        # check for diagonals with negative slope
        pionsConsecutifs = 0
        j = colonne
        for i in range(ligne, self.HAUTEUR+1):
            if j > self.LARGEUR:
                break
            elif (self.obtenirPion(i,j) == self.obtenirPion(ligne, colonne)) and not(self.estVide(ligne,colonne)):
                pionsConsecutifs += 1
            else:
                break
            j += 1 # increment colonneumn when ligne is decremented

        if pionsConsecutifs >= 4:
            count += 1

        if count > 0:
            QuatreEnDiagonale = True

        return QuatreEnDiagonale

    def plateau_en_board(self):
        def str2(pion):
            return {
                pion.VIDE : " ",
                pion.JAUNE : "O",
                pion.ROUGE : "X"
            }.get(pion)
        board=[[" "]*self.LARGEUR for elem in range(self.HAUTEUR)]
        for i in range(self.HAUTEUR):
            for j in range(self.LARGEUR):
                board[i][j]=str2(self.obtenirPion(i+1,j+1))
        return board
    def inverserCouleur(self):
        for i in range(1, self.HAUTEUR+1):
            for j in range(1, self.LARGEUR+1):
                if self.obtenirPion(i,j) == pion.ROUGE:
                    self.placerPion(i,j,pion.JAUNE)
                elif self.obtenirPion(i,j) == pion.JAUNE:
                    self.placerPion(i,j,pion.ROUGE)

    def plateauToKeras(self):
        liste = list()
        for colonne in range(1,self.LARGEUR+1):
            for ligne in range(self.HAUTEUR, 0, -1):
                if self.obtenirPion(ligne, colonne) == pion.VIDE:
                    liste.append(0)
                elif self.obtenirPion(ligne, colonne) == pion.JAUNE:
                    liste.append(1)
                elif self.obtenirPion(ligne, colonne) == pion.ROUGE:
                    liste.append(2)
        liste = np.array(liste)
        liste = np.reshape(liste,(1,42))
        return liste
