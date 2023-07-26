import puissance4
import afficheur
import csv
from joueur import joueurIAMinMax

def creerNouvelleBD():
    old_BD = getBDinitiale()
    new_BD = list()
    for etat_plateau in old_BD:
        convertirListePourNouvelleBD(etat_plateau)
        new_BD.append(etat_plateau)


    with open("data/connect-4-target.data", 'w') as fichier:
        writer = csv.writer(fichier)
        writer.writerows(new_BD)


def getBDinitiale():
    BD = list()

    with open("data/connect-4.data", 'r') as fichier:
        for ligne in fichier.readlines():
            BD.append(ligne.split(","))
    return BD

def getBDfinale():
    BD = list()

    with open("data/connect-4-target.data", 'r') as fichier:
        for ligne in fichier.readlines():
            BD.append(ligne.split(","))
    return BD

def obtenirPlateauAPartirDeListeBD(etat_plateau):
    plateau = puissance4.plateau()
    for i,case in enumerate(etat_plateau):
        if case == "x":
            couleurPion = puissance4.pion.ROUGE
        elif case == "o":
            couleurPion = puissance4.pion.JAUNE
        else:
            couleurPion = puissance4.pion.VIDE

        indice = i + 1
        if (indice % 6 == 0):
            ligne = 1
            colonne = ((indice % 6) // 6)
        else:
            ligne = 7 - (indice % 6)
            colonne = (indice // 6) + 1
        #print("[{},{}]".format(ligne, colonne))

        plateau.placerPion(ligne, colonne, couleurPion)

    return plateau

def convertirListePourNouvelleBD(etat_plateau):
    est_gagnant = etat_plateau.pop()
    #Gagnant si win ou draw
    if (est_gagnant == "loss\n"):
        est_gagnant = False
    else:
        est_gagnant = True;

    plateau = obtenirPlateauAPartirDeListeBD(etat_plateau)
    if not(est_gagnant):
        plateau.inverserCouleur()

    IA = joueurIAMinMax(puissance4.pion.ROUGE)
    # print("lol")
    coupGagnant = IA.choisirColonne(plateau)

    etat_plateau.append(coupGagnant)

def minmax(plateau):
    """Simule temporairement un minmax"""
    return 5;

def extraireInputEtTarget():
    BD = getBDfinale()
    X = list()
    target = list()

    for elemBD in BD:
        target.append(elemBD.pop().rstrip())
        X.append(elemBD)

    return (X, target)

if __name__ == "__main__":

    etat_plateau = getBDinitiale()[100]
    etat_plateau2 =getBDinitiale()[101]
    etat_plateau3 = getBDinitiale()[102]
    etat_plateau.pop()
    etat_plateau2.pop()
    etat_plateau3.pop()

    plateau1 = obtenirPlateauAPartirDeListeBD(etat_plateau)
    plateau2 = obtenirPlateauAPartirDeListeBD(etat_plateau2)
    plateau3 = obtenirPlateauAPartirDeListeBD(etat_plateau3)

    a1 = afficheur.afficheurConsole(plateau1)
    a2 = afficheur.afficheurConsole(plateau2)
    a3 = afficheur.afficheurConsole(plateau3)

    a1.afficherPlateau()
    a2.afficherPlateau()
    a3.afficherPlateau()

    # creerNouvelleBD()

    # (X, target) = extraireInputEtTarget()
    # print(X)
    # print(target)
