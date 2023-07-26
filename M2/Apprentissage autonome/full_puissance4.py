#FrozenLake with "Deep (not so deep) Q-learning"

import random
import numpy
import math
#import tkinter
# import time
#%%

########################################################################
# REGLAGES POUR LA COMPETITION
########################################################################

NB_PARTIES = 1000000  # A adapter (un nombre pair plus grand que 1)
# Pour Ordinateur contre Ordinateur : IA_JAUNE et IA_ROUGE
# Pour Humain contre Ordinateur : IA_ROUGE
# L'IA 15 est actuellement la meilleure
IA_JAUNE = 15  # Numéro de l'IA en compétition (entre 0 et 20 actuellement, mais pas de 11)
IA_ROUGE = 15  # Numéro de l'IA en compétition (entre 0 et 20 actuellement, mais pas de 11)

########################################################################




########################################################################
# CHOIX DE L'AFFICHAGE
########################################################################

MODE_GRAPHIQUE = True  # True : Pour afficher la grille dans une fenêtre et dans la console
                       # False : Pour afficher la grille dans la console exclusivement
TEMPS_CHUTE = 0.1  # 0.1 pour visualiser la chute des pions sinon 0
LARGEUR_GRILLE = 480  # A adapter
ESPACEMENT = LARGEUR_GRILLE / 64  # Espace entre 2 trous de la grille

########################################################################




########################################################################
# CONTRAINTES DU JEU
########################################################################

NB_COLONNES = 7  # Nombre de colonnes de la grille de jeu
NB_LIGNES = 6  # Nombre de lignes de la grille de jeu
ALIGNEMENT = 4  # Nombre de pions à aligner pour gagner

########################################################################

#%%

########################################################################
# FENETRE TKINTER
########################################################################

########################################################################
# Création du widget principal ("parent") : fenetreJeu
########################################################################

fenetreJeu = tkinter.Tk()
fenetreJeu.title("Puissance 4")

########################################################################
# AFFICHAGE DE LA GRILLE ET DES JETONS DANS LA FENETRE TKINTER
########################################################################

def hauteur_grille(r):
    """Hauteur de la grille en fonction du rayon r des trous"""
    return 2*NB_LIGNES*r + (NB_LIGNES + 1)*ESPACEMENT

########################################################################

def rayon():
    """ Rayon des trous de la grille et des pions"""
    return (LARGEUR_GRILLE - (NB_COLONNES + 1)*ESPACEMENT) / (2*NB_COLONNES)

########################################################################

def creation_disque(x, y, r, c, tag):
    """Création d'un disque tag (trou ou jeton), de rayon r et de couleur c à la position (x,y)"""
    identifiant = grille.create_oval(x-r, y-r, x+r, y+r, fill=c, width=0, tags=tag)
    return identifiant

########################################################################

def creation_grille(r):
    """Création de la grille avec des trous de rayon r"""
    ligne = 1
    while ligne <= NB_LIGNES:
        colonne = 1
        while colonne <= NB_COLONNES:
            creation_disque(ESPACEMENT + r + (colonne-1)*(ESPACEMENT + 2*r),
                            ESPACEMENT + r + (ligne-1)*(ESPACEMENT + 2*r),
                            r, 'white', 'trou')
            colonne += 1
        ligne += 1

########################################################################

def creation_jeton(colonne, ligne, c, r):  # Dépend de la grille
    """Création d'un jeton de couleur c et de rayon r à la colonne et à la ligne indiquée"""
    identifiant = creation_disque(colonne*(ESPACEMENT+2*r)-r,
                                  (NB_LIGNES-ligne+1)*(ESPACEMENT+2*r)-r,
                                  r, c, 'jeton')
    return identifiant

########################################################################

def mouvement_jeton(identifiant, r):
    """Mouvement d'un jeton de rayon r"""
    grille.move(identifiant, 0, ESPACEMENT+2*r)

########################################################################

def affiche_grille_fenetre(colonne, ligneSupport, couleur):
    """Affichage du coup joué (avec chute du pion)"""
    ligne = NB_LIGNES
    r = rayon()
    identifiant = creation_jeton(colonne, ligne, couleur, r)
    while ligne > ligneSupport:
        if ligne < NB_LIGNES:
            mouvement_jeton(identifiant, r)
        fenetreJeu.update()
        time.sleep(TEMPS_CHUTE)
        ligne = ligne - 1

########################################################################

def destruction_jetons():
    """Destruction de tous les jetons"""
    grille.delete('jeton')

########################################################################




########################################################################
# Création des widgets "enfants" : grille (Canvas)
########################################################################

grille = tkinter.Canvas(fenetreJeu, width=LARGEUR_GRILLE, height=hauteur_grille(rayon()), background='blue')
creation_grille(rayon())
grille.pack()

########################################################################
# Création des widgets "enfants" : message (Label)
########################################################################

message = tkinter.Label(fenetreJeu)
message.pack()

########################################################################
# Création des widgets "enfants" : scoreJaunes (Label)
########################################################################

scoreJaunes = tkinter.Label(fenetreJeu, text='Jaunes : 0')
scoreJaunes.pack(side='left')

########################################################################
# Création des widgets "enfants" : scoreRouges (Label)
########################################################################

scoreRouges = tkinter.Label(fenetreJeu, text='Rouges : 0')
scoreRouges.pack(side='right')

########################################################################




########################################################################
# AFFICHAGE DES MESSAGES DANS LA FENETRE
########################################################################

def affiche_joueur_qui_commence_fenetre(couleur):
    """Affichage du joueur qui commence dans la fenêtre Tkinter"""
    if couleur == 'yellow':
        message['text'] = 'Les jaunes commencent'
    elif couleur == 'red':
        message['text'] = 'Les rouges commencent'

########################################################################

def affiche_joueur_fenetre(couleur):
    """Affichage du joueur dans la fenêtre Tkinter"""
    if couleur == 'yellow':
        message['text'] = 'Les jaunes jouent'
    elif couleur == 'red':
        message['text'] = 'Les rouges jouent'

########################################################################

def affiche_gagnant_fenetre(couleur):
    """Affichage du gagnant dans la fenêtre Tkinter"""
    if couleur == 'yellow':
        message['text'] = 'Les jaunes gagnent'
    elif couleur == 'red':
        message['text'] = 'Les rouges gagnent'

########################################################################

def affiche_aucun_gagnant_fenetre():
    """Affichage aucun gagnant dans la fenêtre Tkinter"""
    message['text'] = 'Aucun gagnant'

########################################################################

def affiche_victoires_fenetre(victoires):
    """Affichage du nombre de victoires dans la fenêtre Tkinter"""
    [jaunes, rouges, nulles] = victoires
    scoreJaunes['text'] = 'Jaunes : ' + str(jaunes)
    scoreRouges['text'] = 'Rouges : ' + str(rouges)

########################################################################

def efface_message_fenetre():
    """Efface le label message dans la fenêtre Tkinter"""
    message['text'] = ''

########################################################################

def initialise_fenetre(nbParties):
    """ """
    TEMPS_PAUSE = 1
    fenetreJeu.update()
    # Pause en secondes
    time.sleep(TEMPS_PAUSE)
    if nbParties == 2:
        time.sleep(TEMPS_PAUSE*9)  # Pour pouvoir faire une copie écran
    # Dans la fenêtre graphique
    destruction_jetons()
    efface_message_fenetre()
    fenetreJeu.update()
    # Pause en secondes
    time.sleep(TEMPS_PAUSE)

########################################################################


#%%
########################################################################
# AFFICHAGE DE LA GRILLE ET DES JETONS DANS LA CONSOLE
########################################################################

def affiche_grille_console(positions):
    """Affiche la grille dans la console"""
    i = NB_COLONNES*(NB_LIGNES-1)
    while i >= 0:
        print(positions[i:i+NB_COLONNES])
        i = i - NB_COLONNES
    print()

########################################################################




########################################################################
# AFFICHAGE DES MESSAGES DANS LA CONSOLE
########################################################################

def affiche_joueur_qui_commence_console(couleur):
    """Affichage du joueur qui commence dans la console"""
    if couleur == 'yellow':
        print('Les jaunes commencent')
    elif couleur == 'red':
        print('Les rouges commencent')

########################################################################

def affiche_joueur_console(couleur):
    """Affichage du joueur dans la console"""
    if couleur == 'yellow':
        print('Les jaunes jouent')
    elif couleur == 'red':
        print('Les rouges jouent')

########################################################################

def affiche_gagnant_console(couleur):
    """Affichage du gagnant dans la console"""
    if couleur == 'yellow':
        print('Les jaunes gagnent', end='')
    elif couleur == 'red':
        print('Les rouges gagnent', end='')

########################################################################

def affiche_aucun_gagnant_console():
    """Affichage aucun gagnant dans la console"""
    print('Aucun gagnant')

########################################################################

def affiche_victoires_console(victoires):
    """Affichage du nombre de victoires dans la console"""
    [jaunes, rouges, nulles] = victoires
    print('Jaunes : ' + str(jaunes))  # Victoires jaunes
    print('Rouges : ' + str(rouges))  # Victoires rouges
    print('Nulles : ' + str(nulles))  # Parties nulles
    print()

########################################################################




########################################################################
# MOTEUR DU JEU
########################################################################

def initialise_liste_positions():
    """Vide la grille"""
    return [0] * NB_COLONNES*NB_LIGNES

########################################################################

def alignement(somme, nbPions, couleur):
    """Analyse la somme dont il est question dans alignements_pleins() ou alignements_troues() pour détermminer si des pions sont alignés"""
    pionsAlignes = False
    if (couleur == 'yellow' and somme == nbPions) or (couleur == 'red' and somme == -nbPions):
        pionsAlignes = True
    return pionsAlignes

########################################################################

def alignements_pleins(positions, nbPions, couleur):
    """Teste les alignements pleins d'un nombre de pions donné et les retourne sous forme de liste"""
    """
    4 pions alignés : 1111
    3 pions alignés : 111
    2 pions alignés : 11
    1 pion "aligné" : 1
    """
    listeAlignementsPleins = []
    # Vérification des alignements horizontaux
    for j in range(NB_LIGNES):
        for i in range(NB_COLONNES-nbPions+1):
            somme = 0
            for k in range(nbPions):
                somme += positions[NB_COLONNES*j+i+k]
            if alignement(somme, nbPions, couleur):
                listeAlignementsPleins += [i+1,j+1,"H"]
    # Vérification des alignements verticaux
    for j in range(NB_LIGNES-nbPions+1):
        for i in range(NB_COLONNES):
            somme = 0
            for k in range(nbPions):
                somme += positions[NB_COLONNES*j+i+k*NB_COLONNES]
            if alignement(somme, nbPions, couleur):
                listeAlignementsPleins += [i+1,j+1,"V"]
    # Vérification des diagonales montantes
    for j in range(NB_LIGNES-nbPions+1):
        for i in range(NB_COLONNES-nbPions+1):
            somme = 0
            for k in range(nbPions):
                somme += positions[NB_COLONNES*j+i+k*NB_COLONNES+k]
            if alignement(somme, nbPions, couleur):
                listeAlignementsPleins += [i+1,j+1,"DM"]
    # Vérification des diagonales descendantes
    for j in range(nbPions-1, NB_LIGNES):
        for i in range(NB_COLONNES-nbPions+1):
            somme = 0
            for k in range(nbPions):
                somme += positions[NB_COLONNES*j+i-k*NB_COLONNES+k]
            if alignement(somme, nbPions, couleur):
                listeAlignementsPleins += [i+1,j+1,"DD"]
    if listeAlignementsPleins != []:
        listeAlignementsPleins = [nbPions] + listeAlignementsPleins
    return listeAlignementsPleins

########################################################################

def alignements_troues(positions, nbPions, couleur):
    """Teste les alignements troués d'un nombre de pions donné et les retourne sous forme de liste"""
    """
    3 pions alignés : 1110 / 1101 / 1011 / 0111
    2 pions alignés : 110 / 101 / 011
    1 pion "aligné" : 10 / 01
    """
    listeAlignementsTroues = []
    # Vérification des alignements horizontaux
    for j in range(NB_LIGNES):
        for i in range(NB_COLONNES-nbPions):
            somme = 0
            for k in range(nbPions+1):
                somme += positions[NB_COLONNES*j+i+k]
            if alignement(somme, nbPions, couleur):
                listeAlignementsTroues += [i+1,j+1,"H"]
    # Vérification des alignements verticaux
    for j in range(NB_LIGNES-nbPions):
        for i in range(NB_COLONNES):
            somme = 0
            for k in range(nbPions+1):
                somme += positions[NB_COLONNES*j+i+k*NB_COLONNES]
            if alignement(somme, nbPions, couleur):
                listeAlignementsTroues += [i+1,j+1,"V"]
    # Vérification des diagonales montantes
    for j in range(NB_LIGNES-nbPions):
        for i in range(NB_COLONNES-nbPions):
            somme = 0
            for k in range(nbPions+1):
                somme += positions[NB_COLONNES*j+i+k*NB_COLONNES+k]
            if alignement(somme, nbPions, couleur):
                listeAlignementsTroues += [i+1,j+1,"DM"]
    # Vérification des diagonales descendantes
    for j in range(nbPions, NB_LIGNES):
        for i in range(NB_COLONNES-nbPions):
            somme = 0
            for k in range(nbPions+1):
                somme += positions[NB_COLONNES*j+i-k*NB_COLONNES+k]
            if alignement(somme, nbPions, couleur):
                listeAlignementsTroues += [i+1,j+1,"DD"]
    if listeAlignementsTroues != []:
        listeAlignementsTroues = [nbPions] + listeAlignementsTroues
    return listeAlignementsTroues

########################################################################

def grille_pleine(positions):
    """Teste si la grille est pleine"""
    plein = True
    for i in range(NB_LIGNES*NB_COLONNES):
        if positions[i] == 0:
            plein = False
    return plein

########################################################################

def inverse(couleur):
    """ Inverse les couleurs"""
    if couleur == 'yellow':
        couleur = 'red'
    elif couleur == 'red':
        couleur = 'yellow'
    return couleur

########################################################################

def colonne_pleine(positions, colonne):
    """Teste si la colonne indiquée est pleine"""
    plein = True
    position = NB_COLONNES*(NB_LIGNES-1)+colonne-1
    if positions[position] == 0:
        plein = False
    return plein

########################################################################

def jouer(positions, couleur, colonne):
    """Moteur du jeu"""
    if not colonne_pleine(positions, colonne):
        # On remplit la liste des positions
        position = colonne - 1
        ligneSupport = 0
        while positions[position]:
            ligneSupport += 1
            position += NB_COLONNES
        if couleur == 'yellow':
            valeur = 1
        elif couleur == 'red':
            valeur = -1            
        positions[position] = valeur
    # On affiche la grille pour visualiser les positions
    affiche_grille_console(positions)                                   # Affichage Grille
    #if MODE_GRAPHIQUE:
    #    affiche_grille_fenetre(colonne, ligneSupport, couleur)
    return positions



def jouer_bis(positions, couleur, colonne):
    """Moteur du jeu"""
    if not colonne_pleine(positions, colonne):
        # On remplit la liste des positions
        position = colonne - 1
        ligneSupport = 0
        while positions[position]:
            ligneSupport += 1
            position += NB_COLONNES
        if couleur == 'yellow':
            valeur = 1
        elif couleur == 'red':
            valeur = -1            
        positions[position] = valeur
    return positions

########################################################################

def fin_partie(positions, couleur, victoires):
    """ Test de fin de partie"""
    [jaunes, rouges, nulles] = victoires
    # On teste si la partie est finie
    fin = False
    if alignements_pleins(positions, ALIGNEMENT, couleur):
        fin = True
        if couleur == 'yellow':
            jaunes += 1
        elif couleur == 'red':
            rouges += 1
        # On affiche le gagnant
        affiche_gagnant_console(couleur)
        nbCoups = analyse_victoire(positions)
        print(" en", nbCoups, "coups")
        #if MODE_GRAPHIQUE:
        #    affiche_gagnant_fenetre(couleur)
    elif grille_pleine(positions):
        fin = True
        nulles += 1
        # On affiche aucun gagnant
        affiche_aucun_gagnant_console()
        #if MODE_GRAPHIQUE:
        #    affiche_aucun_gagnant_fenetre()
    else:
        couleur = inverse(couleur)
        # On affiche qui doit jouer
        affiche_joueur_console(couleur)
        #if MODE_GRAPHIQUE:
        #    affiche_joueur_fenetre(couleur)
    victoires = [jaunes, rouges, nulles]
    return fin, couleur, victoires

def fin_partie_bis(positions, couleur, victoires):
    """ Test de fin de partie"""
    [jaunes, rouges, nulles] = victoires
    # On teste si la partie est finie
    fin = False
    if alignements_pleins(positions, ALIGNEMENT, couleur):
        fin = True
        if couleur == 'yellow':
            jaunes += 1
        elif couleur == 'red':
            rouges += 1
    elif grille_pleine(positions):
        fin = True
        nulles += 1
    else:
        couleur = inverse(couleur)
    victoires = [jaunes, rouges, nulles]
    return fin, couleur, victoires

########################################################################




########################################################################
# ANALYSE
########################################################################

def analyse_victoire(positions):
    """Analyse la victoire"""
    # Nombre de coups du gagnant
    nbPositionsPleines = NB_LIGNES*NB_COLONNES
    for i in range(NB_LIGNES*NB_COLONNES):
        if positions[i] == 0:
            nbPositionsPleines -= 1
    return math.ceil(nbPositionsPleines/2)  ## Arrondi à l'entier supérieur

########################################################################

#%%
########################################################################
# STRATEGIES DE JEU
########################################################################

def jouer_ordi_hasard(positions, couleur):
    """L'ordinateur joue au hasard"""
    colonne = random.randint(1, NB_COLONNES)
    while colonne_pleine(positions, colonne):
        colonne = random.randint(1, NB_COLONNES)
    return jouer(positions, couleur, colonne)

########################################################################

def poids_cases():
    """Calcule le poids des cases en fonction de la dimension de la grille et du nombre de pions à aligner pour gagner"""
    """[3,4,5,7,5,4,3,4,6,8,10,8,6,4,5,8,11,13,11,8,5,5,8,11,13,11,8,5,4,6,8,10,8,6,4,3,4,5,7,5,4,3] pour une grille 7x6 avec 4 pions à aligner"""
    poids = [0] * NB_COLONNES*NB_LIGNES
    # Sur les horizontales
    for j in range(NB_LIGNES):
        for i in range(NB_COLONNES-ALIGNEMENT+1):
            for k in range(ALIGNEMENT):
                poids[NB_COLONNES*j+i+k] += 1
    # Sur les verticales
    for j in range(NB_LIGNES-ALIGNEMENT+1):
        for i in range(NB_COLONNES):
            for k in range(ALIGNEMENT):
                poids[NB_COLONNES*j+i+k*NB_COLONNES] += 1
    # Sur les diagonales montantes
    for j in range(NB_LIGNES-ALIGNEMENT+1):
        for i in range(NB_COLONNES-ALIGNEMENT+1):
            for k in range(ALIGNEMENT):
                poids[NB_COLONNES*j+i+k*NB_COLONNES+k] += 1
    # Sur les diagonales descendantes
    for j in range(ALIGNEMENT-1, NB_LIGNES):
        for i in range(NB_COLONNES-ALIGNEMENT+1):
            for k in range(ALIGNEMENT):
                poids[NB_COLONNES*j+i-k*NB_COLONNES+k] += 1
    return poids

########################################################################

def liste_indices_maximum(liste):
    """Renvoie les indices des maximums d'une liste"""
    maxi = max(liste)
    indices = []
    for i in range(len(liste)):
        if liste[i] == maxi:
            indices += [i]
    return indices

########################################################################

def jouer_ordi_poids_cases(positions, couleur):
    """L'ordinateur joue en ne tenant compte que du poids des cases de la grille potentiellement victorieuses"""
    poidsCases = poids_cases()
    poidsColonnes = [0] * NB_COLONNES
    for colonne in range(1, NB_COLONNES + 1):
        if not colonne_pleine(positions, colonne):
            position = colonne - 1
            while positions[position]:
                position += NB_COLONNES
            poidsColonnes[colonne - 1] += poidsCases[position]
        else:
            poidsColonnes[colonne - 1] += 0
    indicesPoidsMaximum = liste_indices_maximum(poidsColonnes)
    # Si plusieurs colonnes sont possibles (même poids), on tire au hasard une colonne
    colonne = 1 + random.choice(indicesPoidsMaximum)
    return jouer(positions, couleur, colonne)

def jouer_ordi_poids_cases_bis(positions, couleur):
    """L'ordinateur joue en ne tenant compte que du poids des cases de la grille potentiellement victorieuses"""
    poidsCases = poids_cases()
    poidsColonnes = [0] * NB_COLONNES
    for colonne in range(1, NB_COLONNES + 1):
        if not colonne_pleine(positions, colonne):
            position = colonne - 1
            while positions[position]:
                position += NB_COLONNES
            poidsColonnes[colonne - 1] += poidsCases[position]
        else:
            poidsColonnes[colonne - 1] += 0
    indicesPoidsMaximum = liste_indices_maximum(poidsColonnes)
    # Si plusieurs colonnes sont possibles (même poids), on tire au hasard une colonne
    colonne = 1 + random.choice(indicesPoidsMaximum)
    return jouer_bis(positions, couleur, colonne)

########################################################################
""" ANCIENNE VERSION (SANS HASARD EN CAS D'EGALITE DE POIDS)
def jouer_ordi_poids_cases(positions, couleur):
    "L'ordinateur joue en ne tenant compte que du poids des cases de la grille (7x6) potentiellement victorieuses"
    POIDS_POSITIONS = [3,4,5,7,5,4,3,4,6,8,10,8,6,4,5,8,11,13,11,8,5,5,8,11,13,11,8,5,4,6,8,10,8,6,4,3,4,5,7,5,4,3]
    poidsColonne = [0] * NB_COLONNES
    for colonne in range(1, NB_COLONNES + 1):
        if not colonne_pleine(positions, colonne):
            position = colonne - 1
            while positions[position]:
                position += NB_COLONNES
            poidsColonne[colonne - 1] = POIDS_POSITIONS[position]
        else:
            poidsColonne[colonne - 1] = 0
        colonne = poidsColonne.index(max(poidsColonne)) + 1
    return jouer(positions, couleur, colonne)
"""
########################################################################

def position(colonne, ligne):
    """Déduit d'une position dans la grille une position dans la liste positions[]"""
    position = (colonne-1) + (ligne-1)*NB_COLONNES
    return position

########################################################################

def colonne_extraite(position):
    """Déduit d'une position dans la grille la colonne correspondante"""
    colonne = position % NB_COLONNES + 1
    return colonne

########################################################################

def position_potentielle(positions, colonne, ligne):
    """Teste si une position est possible (case vide et support pour soutenir le pion)"""
    test = False
    if colonne >= 1 and colonne <= NB_COLONNES and ligne >= 1 and ligne <= NB_LIGNES:
        if positions[position(colonne, ligne)] == 0:  # Position libre
            test = True
            if ligne > 1:
                if positions[position(colonne, ligne - 1)] == 0:  # Ligne support inexistante
                    test = False
    return test

########################################################################

def meilleure_position(positionsPotentielles):
    """Détermine la meilleure position en s'appuyant sur le poids des cases"""
    # Calcule le poids des cases
    poidsCases = poids_cases()
    # Détermine le poids des positions potentielles
    poidsPositionsPotentielles = []
    for i in range(len(positionsPotentielles)):
        poidsPositionsPotentielles += [poidsCases[positionsPotentielles[i]]]
    # Détermine les indices du poids maximum dans la liste ci-dessus
    indicesPoidsMaximum = liste_indices_maximum(poidsPositionsPotentielles)
    # Extrait les meilleures positions potentielles (celles qui ont un poids maximum)
    meilleuresPositionsPotentielles = []
    for i in range(len(indicesPoidsMaximum)):
        meilleuresPositionsPotentielles += [positionsPotentielles[indicesPoidsMaximum[i]]]
    # Si plusieurs positions sont possibles (même poids), on tire au hasard une position
    return random.choice(meilleuresPositionsPotentielles)

########################################################################
""" ANCIENNE VERSION (SANS HASARD EN CAS D'EGALITE DE POIDS)
def meilleure_position(positionsPotentielles):
    ""
    POIDS_POSITIONS = [3,4,5,7,5,4,3,4,6,8,10,8,6,4,5,8,11,13,11,8,5,5,8,11,13,11,8,5,4,6,8,10,8,6,4,3,4,5,7,5,4,3]
    poidsMax = 0
    longueurListe = len(positionsPotentielles)
    for i in range(longueurListe):
        if POIDS_POSITIONS[positionsPotentielles[i]] > poidsMax:
            poidsMax = POIDS_POSITIONS[positionsPotentielles[i]]
            iMax = i
    return positionsPotentielles[iMax]
"""
########################################################################

def positions_potentielles(positions, listeAlignementsTroues):
    """Retourne une colonne où jouer à partir de l'ensemble des positions potentielles"""
    positionsPotentielles = []
    if listeAlignementsTroues != []:
        nbPions = listeAlignementsTroues[0]
        for i in range(0, len(listeAlignementsTroues) // 3):
            c = listeAlignementsTroues[1 + 3*i] # Colonne
            l = listeAlignementsTroues[2 + 3*i] # Ligne
            d = listeAlignementsTroues[3 + 3*i] # Direction
            if d == "H": # Horizontal
                for j in range(nbPions + 1):
                    if position_potentielle(positions, c + j, l):
                        positionsPotentielles += [position(c + j, l)]
            if d == "V": # Vertical
                if position_potentielle(positions, c, l + nbPions):
                    positionsPotentielles += [position(c, l + nbPions)]
            if d == "DM": # Diagonale Montante
                for j in range(nbPions + 1):
                    if position_potentielle(positions, c + j, l + j):
                        positionsPotentielles += [position(c + j, l + j)]
            if d == "DD": # Diagonale Descendante
                for j in range(nbPions + 1):
                    if position_potentielle(positions, c + j, l - j):
                        positionsPotentielles += [position(c + j, l - j)]
    colonne = -1
    if len(positionsPotentielles) > 0:
        colonne = colonne_extraite(meilleure_position(positionsPotentielles))
    return colonne

########################################################################

def priorite_pleine(positions, nbPions, couleur):
    """Retourne une colonne où jouer"""
    listeAlignementsPleins = alignements_pleins(positions, nbPions-1, couleur)
    return positions_potentielles(positions, listeAlignementsPleins)

########################################################################

def priorite_trouee(positions, nbPions, couleur):
    """Retourne une colonne où jouer"""
    listeAlignementsTroues = alignements_troues(positions, nbPions-1, couleur)
    return positions_potentielles(positions, listeAlignementsTroues)

########################################################################




########################################################################
# LISTE DES IA
########################################################################

def jouer_ordi_ia0(positions, couleur):
    """IA0 joue"""
    return jouer_ordi_hasard(positions, couleur)                     # H   : L'IA joue au hasard

########################################################################

def jouer_ordi_ia1(positions, couleur):
    """IA1 joue"""
    return jouer_ordi_poids_cases(positions, couleur)                # PH  : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia2(positions, couleur):                              # Priorité Alignements Pleins
    """IA2 joue"""
    colA4PH = priorite_pleine(positions, 4, couleur)
    colA3PH = priorite_pleine(positions, 3, couleur)
    colA2PH = priorite_pleine(positions, 2, couleur)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colA3PH != -1: return jouer(positions, couleur, colA3PH)    # A3PH : L'IA essaye d'aligner 3 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia3(positions, couleur):                              # Priorité Alignements Pleins
    """IA3 joue"""
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_pleine(positions, 4, couleurAdversaire)
    colB3PH = priorite_pleine(positions, 3, couleurAdversaire)
    colB2PH = priorite_pleine(positions, 2, couleurAdversaire)
    if colB4PH != -1: return jouer(positions, couleur, colB4PH)      # B4PH : L'IA essaye en priorité d'empêcher l'adversaire d'aligner 4 pions
    elif colB3PH != -1: return jouer(positions, couleur, colB3PH)    # B3PH : L'IA essaye d'empêcher l'adversaire d'aligner 3 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia4(positions, couleur):                              # Priorité Alignements Pleins
    """IA4 joue"""
    colA4PH = priorite_pleine(positions, 4, couleur)
    colA3PH = priorite_pleine(positions, 3, couleur)
    colA2PH = priorite_pleine(positions, 2, couleur)
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_pleine(positions, 4, couleurAdversaire)
    colB3PH = priorite_pleine(positions, 3, couleurAdversaire)
    colB2PH = priorite_pleine(positions, 2, couleurAdversaire)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colB4PH != -1: return jouer(positions, couleur, colB4PH)    # B4PH : L'IA essaye d'empêcher l'adversaire d'aligner 4 pions
    elif colA3PH != -1: return jouer(positions, couleur, colA3PH)    # A3PH : L'IA essaye d'aligner 3 pions
    elif colB3PH != -1: return jouer(positions, couleur, colB3PH)    # B3PH : L'IA essaye d'empêcher l'adversaire d'aligner 3 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia5(positions, couleur):                              # Priorité Alignements Pleins
    """IA5 joue"""
    colA4PH = priorite_pleine(positions, 4, couleur)
    colA3PH = priorite_pleine(positions, 3, couleur)
    colA2PH = priorite_pleine(positions, 2, couleur)
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_pleine(positions, 4, couleurAdversaire)
    colB3PH = priorite_pleine(positions, 3, couleurAdversaire)
    colB2PH = priorite_pleine(positions, 2, couleurAdversaire)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colB4PH != -1: return jouer(positions, couleur, colB4PH)    # B4PH : L'IA essaye d'empêcher l'adversaire d'aligner 4 pions
    elif colA3PH != -1: return jouer(positions, couleur, colA3PH)    # A3PH : L'IA essaye d'aligner 3 pions
    elif colB3PH != -1: return jouer(positions, couleur, colB3PH)    # B3PH : L'IA essaye d'empêcher l'adversaire d'aligner 3 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia6(positions, couleur):                              # Priorité Alignements Pleins
    """IA6 joue"""
    colA4PH = priorite_pleine(positions, 4, couleur)
    colA3PH = priorite_pleine(positions, 3, couleur)
    colA2PH = priorite_pleine(positions, 2, couleur)
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_pleine(positions, 4, couleurAdversaire)
    colB3PH = priorite_pleine(positions, 3, couleurAdversaire)
    colB2PH = priorite_pleine(positions, 2, couleurAdversaire)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colB4PH != -1: return jouer(positions, couleur, colB4PH)    # B4PH : L'IA essaye d'empêcher l'adversaire d'aligner 4 pions
    elif colB3PH != -1: return jouer(positions, couleur, colB3PH)    # B3PH : L'IA essaye d'empêcher l'adversaire d'aligner 3 pions
    elif colA3PH != -1: return jouer(positions, couleur, colA3PH)    # A3PH : L'IA essaye d'aligner 3 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia7(positions, couleur):                              # Priorité Alignements Pleins
    """IA7 joue"""
    colA4PH = priorite_pleine(positions, 4, couleur)
    colA3PH = priorite_pleine(positions, 3, couleur)
    colA2PH = priorite_pleine(positions, 2, couleur)
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_pleine(positions, 4, couleurAdversaire)
    colB3PH = priorite_pleine(positions, 3, couleurAdversaire)
    colB2PH = priorite_pleine(positions, 2, couleurAdversaire)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colB4PH != -1: return jouer(positions, couleur, colB4PH)    # B4PH : L'IA essaye d'empêcher l'adversaire d'aligner 4 pions
    elif colB3PH != -1: return jouer(positions, couleur, colB3PH)    # B3PH : L'IA essaye d'empêcher l'adversaire d'aligner 3 pions
    elif colA3PH != -1: return jouer(positions, couleur, colA3PH)    # A3PH : L'IA essaye d'aligner 3 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia8(positions, couleur):                              # Priorité Alignements Pleins
    """IA8 joue"""
    colA4PH = priorite_pleine(positions, 4, couleur)
    colA2PH = priorite_pleine(positions, 2, couleur)
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_pleine(positions, 4, couleurAdversaire)
    colB2PH = priorite_pleine(positions, 2, couleurAdversaire)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colB4PH != -1: return jouer(positions, couleur, colB4PH)    # B4PH : L'IA essaye d'empêcher l'adversaire d'aligner 4 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia9(positions, couleur):                              # Priorité Alignements Pleins
    """IA9 joue"""
    colA4PH = priorite_pleine(positions, 4, couleur)
    colA2PH = priorite_pleine(positions, 2, couleur)
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_pleine(positions, 4, couleurAdversaire)
    colB2PH = priorite_pleine(positions, 2, couleurAdversaire)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colB4PH != -1: return jouer(positions, couleur, colB4PH)    # B4PH : L'IA essaye d'empêcher l'adversaire d'aligner 4 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia10(positions, couleur):                             # Priorité Alignements Pleins
    """IA10 joue"""
    colA4PH = priorite_pleine(positions, 4, couleur)
    colA3PH = priorite_pleine(positions, 3, couleur)
    colA2PH = priorite_pleine(positions, 2, couleur)
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_pleine(positions, 4, couleurAdversaire)
    colB3PH = priorite_pleine(positions, 3, couleurAdversaire)
    colB2PH = priorite_pleine(positions, 2, couleurAdversaire)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colB4PH != -1: return jouer(positions, couleur, colB4PH)    # B4PH : L'IA essaye d'empêcher l'adversaire d'aligner 4 pions
    elif colB3PH != -1: return jouer(positions, couleur, colB3PH)    # B3PH : L'IA essaye d'empêcher l'adversaire d'aligner 3 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    elif colA3PH != -1: return jouer(positions, couleur, colA3PH)    # A3PH : L'IA essaye d'aligner 3 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

# Pas d'IA11

########################################################################

def jouer_ordi_ia12(positions, couleur):                            # Priorité Alignements Troués
    """IA2 joue"""
    colA4PH = priorite_trouee(positions, 4, couleur)
    colA3PH = priorite_trouee(positions, 3, couleur)
    colA2PH = priorite_trouee(positions, 2, couleur)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colA3PH != -1: return jouer(positions, couleur, colA3PH)    # A3PH : L'IA essaye d'aligner 3 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia13(positions, couleur):                             # Priorité Alignements Troués
    """IA3 joue"""
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_trouee(positions, 4, couleurAdversaire)
    colB3PH = priorite_trouee(positions, 3, couleurAdversaire)
    colB2PH = priorite_trouee(positions, 2, couleurAdversaire)
    if colB4PH != -1: return jouer(positions, couleur, colB4PH)      # B4PH : L'IA essaye en priorité d'empêcher l'adversaire d'aligner 4 pions
    elif colB3PH != -1: return jouer(positions, couleur, colB3PH)    # B3PH : L'IA essaye d'empêcher l'adversaire d'aligner 3 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia14(positions, couleur):                             # Priorité Alignements Troués
    """IA4 joue"""
    colA4PH = priorite_trouee(positions, 4, couleur)
    colA3PH = priorite_trouee(positions, 3, couleur)
    colA2PH = priorite_trouee(positions, 2, couleur)
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_trouee(positions, 4, couleurAdversaire)
    colB3PH = priorite_trouee(positions, 3, couleurAdversaire)
    colB2PH = priorite_trouee(positions, 2, couleurAdversaire)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colB4PH != -1: return jouer(positions, couleur, colB4PH)    # B4PH : L'IA essaye d'empêcher l'adversaire d'aligner 4 pions
    elif colA3PH != -1: return jouer(positions, couleur, colA3PH)    # A3PH : L'IA essaye d'aligner 3 pions
    elif colB3PH != -1: return jouer(positions, couleur, colB3PH)    # B3PH : L'IA essaye d'empêcher l'adversaire d'aligner 3 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia15(positions, couleur):                             # Priorité Alignements Troués
    """IA5 joue"""
    colA4PH = priorite_trouee(positions, 4, couleur)
    colA3PH = priorite_trouee(positions, 3, couleur)
    colA2PH = priorite_trouee(positions, 2, couleur)
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_trouee(positions, 4, couleurAdversaire)
    colB3PH = priorite_trouee(positions, 3, couleurAdversaire)
    colB2PH = priorite_trouee(positions, 2, couleurAdversaire)
    if colA4PH != -1: return jouer_bis(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colB4PH != -1: return jouer_bis(positions, couleur, colB4PH)    # B4PH : L'IA essaye d'empêcher l'adversaire d'aligner 4 pions
    elif colA3PH != -1: return jouer_bis(positions, couleur, colA3PH)    # A3PH : L'IA essaye d'aligner 3 pions
    elif colB3PH != -1: return jouer_bis(positions, couleur, colB3PH)    # B3PH : L'IA essaye d'empêcher l'adversaire d'aligner 3 pions
    elif colB2PH != -1: return jouer_bis(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    elif colA2PH != -1: return jouer_bis(positions, couleur, colA2PH)
    else: return jouer_ordi_poids_cases_bis(positions, couleur)# PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia16(positions, couleur):                             # Priorité Alignements Troués
    """IA6 joue"""
    colA4PH = priorite_trouee(positions, 4, couleur)
    colA3PH = priorite_trouee(positions, 3, couleur)
    colA2PH = priorite_trouee(positions, 2, couleur)
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_trouee(positions, 4, couleurAdversaire)
    colB3PH = priorite_trouee(positions, 3, couleurAdversaire)
    colB2PH = priorite_trouee(positions, 2, couleurAdversaire)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colB4PH != -1: return jouer(positions, couleur, colB4PH)    # B4PH : L'IA essaye d'empêcher l'adversaire d'aligner 4 pions
    elif colB3PH != -1: return jouer(positions, couleur, colB3PH)    # B3PH : L'IA essaye d'empêcher l'adversaire d'aligner 3 pions
    elif colA3PH != -1: return jouer(positions, couleur, colA3PH)    # A3PH : L'IA essaye d'aligner 3 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia17(positions, couleur):                             # Priorité Alignements Troués
    """IA7 joue"""
    colA4PH = priorite_trouee(positions, 4, couleur)
    colA3PH = priorite_trouee(positions, 3, couleur)
    colA2PH = priorite_trouee(positions, 2, couleur)
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_trouee(positions, 4, couleurAdversaire)
    colB3PH = priorite_trouee(positions, 3, couleurAdversaire)
    colB2PH = priorite_trouee(positions, 2, couleurAdversaire)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colB4PH != -1: return jouer(positions, couleur, colB4PH)    # B4PH : L'IA essaye d'empêcher l'adversaire d'aligner 4 pions
    elif colB3PH != -1: return jouer(positions, couleur, colB3PH)    # B3PH : L'IA essaye d'empêcher l'adversaire d'aligner 3 pions
    elif colA3PH != -1: return jouer(positions, couleur, colA3PH)    # A3PH : L'IA essaye d'aligner 3 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia18(positions, couleur):                             # Priorité Alignements Troués
    """IA8 joue"""
    colA4PH = priorite_trouee(positions, 4, couleur)
    colA2PH = priorite_trouee(positions, 2, couleur)
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_trouee(positions, 4, couleurAdversaire)
    colB2PH = priorite_trouee(positions, 2, couleurAdversaire)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colB4PH != -1: return jouer(positions, couleur, colB4PH)    # B4PH : L'IA essaye d'empêcher l'adversaire d'aligner 4 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia19(positions, couleur):                             # Priorité Alignements Troués
    """IA9 joue"""
    colA4PH = priorite_trouee(positions, 4, couleur)
    colA2PH = priorite_trouee(positions, 2, couleur)
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_trouee(positions, 4, couleurAdversaire)
    colB2PH = priorite_trouee(positions, 2, couleurAdversaire)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colB4PH != -1: return jouer(positions, couleur, colB4PH)    # B4PH : L'IA essaye d'empêcher l'adversaire d'aligner 4 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia20(positions, couleur):                             # Priorité Alignements Troués
    """IA10 joue"""
    colA4PH = priorite_trouee(positions, 4, couleur)
    colA3PH = priorite_trouee(positions, 3, couleur)
    colA2PH = priorite_trouee(positions, 2, couleur)
    couleurAdversaire = inverse(couleur)
    colB4PH = priorite_trouee(positions, 4, couleurAdversaire)
    colB3PH = priorite_trouee(positions, 3, couleurAdversaire)
    colB2PH = priorite_trouee(positions, 2, couleurAdversaire)
    if colA4PH != -1: return jouer(positions, couleur, colA4PH)      # A4PH : L'IA essaye en priorité d'aligner 4 pions
    elif colB4PH != -1: return jouer(positions, couleur, colB4PH)    # B4PH : L'IA essaye d'empêcher l'adversaire d'aligner 4 pions
    elif colB3PH != -1: return jouer(positions, couleur, colB3PH)    # B3PH : L'IA essaye d'empêcher l'adversaire d'aligner 3 pions
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    elif colA3PH != -1: return jouer(positions, couleur, colA3PH)    # A3PH : L'IA essaye d'aligner 3 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)    # A2PH : L'IA essaye d'aligner 2 pions
    else: return jouer_ordi_poids_cases(positions, couleur)          # PH   : L'IA joue dans la case qui a le plus de poids

########################################################################

def jouer_ordi_ia(positions, couleur, ia):
    """L'IA choisie joue"""
    if ia == 0: positions = jouer_ordi_ia0(positions, couleur)
    elif ia == 1: positions = jouer_ordi_ia1(positions, couleur)
    
    elif ia == 2: positions = jouer_ordi_ia2(positions, couleur)
    elif ia == 3: positions = jouer_ordi_ia3(positions, couleur)
    elif ia == 4: positions = jouer_ordi_ia4(positions, couleur)
    elif ia == 5: positions = jouer_ordi_ia5(positions, couleur)
    elif ia == 6: positions = jouer_ordi_ia6(positions, couleur)
    elif ia == 7: positions = jouer_ordi_ia7(positions, couleur)
    elif ia == 8: positions = jouer_ordi_ia8(positions, couleur)
    elif ia == 9: positions = jouer_ordi_ia9(positions, couleur)
    elif ia == 10: positions = jouer_ordi_ia10(positions, couleur)
    
    elif ia == 12: positions = jouer_ordi_ia12(positions, couleur)
    elif ia == 13: positions = jouer_ordi_ia13(positions, couleur)
    elif ia == 14: positions = jouer_ordi_ia14(positions, couleur)
    elif ia == 15: positions = jouer_ordi_ia15(positions, couleur)
    elif ia == 16: positions = jouer_ordi_ia16(positions, couleur)
    elif ia == 17: positions = jouer_ordi_ia17(positions, couleur)
    elif ia == 18: positions = jouer_ordi_ia18(positions, couleur)
    elif ia == 19: positions = jouer_ordi_ia19(positions, couleur)
    elif ia == 20: positions = jouer_ordi_ia20(positions, couleur)
    return positions

########################################################################


#%%
epsilon=0.1
gamma=0.9
alpha=0.5
alpha_reg=0.9999

def sigmoid(x):
    if x>100:
        returnValue=1
    elif x<-100:
        returnValue=-1
    else:
        returnValue=(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    return returnValue
        
class NN:
    def __init__(self,sizeInput,sizeHiddenLayer,sizeOutput):
        self.sizeInput=sizeInput
        self.sizeHiddenLayer=sizeHiddenLayer
        self.sizeOutput=sizeOutput

        #below are the weights
        self.HiddenLayerEntryWeights=numpy.zeros([sizeHiddenLayer,sizeInput])
        self.LastLayerEntryWeights=numpy.zeros([sizeOutput,sizeHiddenLayer])

        #random initialization
        for i in range(0,sizeHiddenLayer):
            for j in range(0,sizeInput):
                self.HiddenLayerEntryWeights[i,j]=random.uniform(-0.1,0.1)
                
        for i in range(0,sizeOutput):
            for j in range(0,sizeHiddenLayer):
                self.LastLayerEntryWeights[i,j]=random.uniform(-0.1,0.1)

        self.HiddenLayerEntryDeltas=numpy.zeros(sizeHiddenLayer)
        self.LastLayerEntryDeltas=numpy.zeros(sizeOutput)

        self.HiddenLayerOutput=numpy.zeros(sizeHiddenLayer)
        self.LastLayerOutput=numpy.zeros(sizeOutput)

    def output(self,x):
        for i in range(0, self.sizeHiddenLayer):
            self.HiddenLayerOutput[i]=sigmoid(numpy.dot(self.HiddenLayerEntryWeights[i],x))
        for i in range(0, self.sizeOutput):
            self.LastLayerOutput[i]= \
            sigmoid(numpy.dot(self.LastLayerEntryWeights[i],self.HiddenLayerOutput))

    def retropropagation(self,x,y,actionIndex):
        self.output(x)

        #deltas computation
        self.LastLayerEntryDeltas[actionIndex]=2*(self.LastLayerOutput[actionIndex]-y)* \
            (1+self.LastLayerOutput[actionIndex])*(1-self.LastLayerOutput[actionIndex])

        for i in range(0,self.sizeHiddenLayer):
            #here usually you need a sum
            self.HiddenLayerEntryDeltas[i]=self.LastLayerEntryDeltas[actionIndex]* \
            (1+self.HiddenLayerOutput[i])*(1-self.HiddenLayerOutput[i])*self.LastLayerEntryWeights[actionIndex,i]

        #weights update
        for i in range(0,self.sizeHiddenLayer):
            self.LastLayerEntryWeights[actionIndex,i]-=alpha*self.LastLayerEntryDeltas[actionIndex]* \
            self.HiddenLayerOutput[i]

        for i in range(0,self.sizeHiddenLayer):
            for j in range(0,self.sizeInput):
                self.HiddenLayerEntryWeights[i,j]-=alpha*self.HiddenLayerEntryDeltas[i]*x[j]

        
nbCells=6*7
sizeInput=nbCells
sizeHiddenLayer=100
sizeOutput=7
myNN = NN(sizeInput, sizeHiddenLayer, sizeOutput)
cpt_victoire=[0,0,0]
next_cpt_victoire=[0,0,0]


#%% apprentissage myNN vs ia.py
alpha=0.5
nbEpisodes=50000

for i in range(0, nbEpisodes):
    alpha=alpha*alpha_reg
    
    if i%100==0:
        print("episode: "+str(i))
        print("  -Score:"+str(cpt_victoire))
        cpt_victoire=[0,0,0]

    state=initialise_liste_positions()

    #--------------------
    tirage_joueur=random.randint(1,2)
    if tirage_joueur==1:
        joueur_actuel="yellow"
    else :
        joueur_actuel="red"
    #--------------------

    
    firstState=state
    endOfEpisode = False
    nbSteps=0
    reward=0

    while not endOfEpisode:
        if joueur_actuel=="yellow":
            if random.uniform(0, 1) < epsilon:
                action = random.randint(1, 7)
                while colonne_pleine(state, action):
                    action = random.randint(1, 7)
            else:
                x=state
                myNN.output(x)
                action = numpy.argmax(myNN.LastLayerOutput)+1
                cpt_action=0
                while colonne_pleine(state, action):
                    cpt_action+=1
                    action=numpy.argsort(-myNN.LastLayerOutput)[cpt_action]+1
                    

            next_state=jouer_bis(state, joueur_actuel, action)  
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie_bis(next_state, joueur_actuel, cpt_victoire) 
            if endOfEpisode==True:
                reward=1
            
        elif joueur_actuel=="red":
            jouer_ordi_ia(state, "red", 15)
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie_bis(state, joueur_actuel, cpt_victoire)
            if endOfEpisode==True:
                reward=-1

        
        x=state
            
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)

        target = reward+gamma*next_max
   
        x=state
        myNN.retropropagation(x,target,action-1)
  
        state = next_state
        joueur_actuel=joueur_suivant    
    
        cpt_victoire=next_cpt_victoire
        nbSteps=nbSteps+1
        
print("end of learning period")

#%% apprentissage myNN vs ia rand


alpha=0.5
nbEpisodes=50000

for i in range(0, nbEpisodes):
    alpha=alpha*alpha_reg
    
    if i%100==0:
        print("episode: "+str(i))
        print("  -Score:"+str(cpt_victoire))
        cpt_victoire=[0,0,0]

    state=initialise_liste_positions()
    
    #--------------------
    tirage_joueur=random.randint(1,2)
    if tirage_joueur==1:
        joueur_actuel="yellow"
    else :
        joueur_actuel="red"
    #--------------------
    
    firstState=state
    endOfEpisode = False
    nbSteps=0

    while not endOfEpisode:
        if joueur_actuel=="yellow":
            if random.uniform(0, 1) < epsilon:
                action = random.randint(1, 7)
                while colonne_pleine(state, action):
                    action = random.randint(1, 7)
            else:
                x=state
                myNN.output(x)
                action = numpy.argmax(myNN.LastLayerOutput)+1
                cpt_action=0
                while colonne_pleine(state, action):
                    cpt_action+=1
                    action=numpy.argsort(-myNN.LastLayerOutput)[cpt_action]+1
                    

            next_state=jouer_bis(state, joueur_actuel, action)       
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie_bis(next_state, joueur_actuel, cpt_victoire) 
            
        elif joueur_actuel=="red":
            action = random.randint(1, 7)
            while colonne_pleine(state, action):
                action = random.randint(1, 7)
            
            next_state=jouer_bis(state, joueur_actuel, action)
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie_bis(state, joueur_actuel, cpt_victoire)
            

        if (next_cpt_victoire[0]-cpt_victoire[0])==1:
            reward=1
        elif (next_cpt_victoire[1]-cpt_victoire[1])==1:
            reward=-1
        else :
            reward=0

        x=state
            
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)

        target = reward+gamma*next_max
   
        x=state
        myNN.retropropagation(x,target,action-1)
  
        state = next_state
        joueur_actuel=joueur_suivant    
    
        cpt_victoire=next_cpt_victoire
        nbSteps=nbSteps+1
print("end of learning period")

#%% apprentissange myNN vs myNN mise à jour de l'adversaire 
# toutes les 1OOO ittérations)
myNN_past=myNN

alpha=0.5
nbEpisodes=50000

for i in range(0, nbEpisodes):
    if i%1000==0:
        myNN_past=myNN
    
    alpha=alpha*alpha_reg
    
    if i%100==0:
        print("episode: "+str(i))
        print("  -Score:"+str(cpt_victoire))
        cpt_victoire=[0,0,0]

    state=initialise_liste_positions()
    
    #--------------------
    tirage_joueur=random.randint(1,2)
    if tirage_joueur==1:
        joueur_actuel="yellow"
    else :
        joueur_actuel="red"
    #--------------------

    
    firstState=state
    endOfEpisode = False
    nbSteps=0

    while not endOfEpisode:
        if joueur_actuel=="yellow":
            if random.uniform(0, 1) < epsilon:
                action = random.randint(1, 7)
                while colonne_pleine(state, action):
                    action = random.randint(1, 7)
            else:
                x=state
                myNN.output(x)
                action = numpy.argmax(myNN.LastLayerOutput)+1
                cpt_action=0
                while colonne_pleine(state, action):
                    cpt_action+=1
                    action=numpy.argsort(-myNN.LastLayerOutput)[cpt_action]+1
                    

            next_state=jouer_bis(state, joueur_actuel, action)       
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie_bis(next_state, joueur_actuel, cpt_victoire) 
            
        elif joueur_actuel=="red":
            x=state
            myNN_past.output(x)
            action = numpy.argmax(myNN_past.LastLayerOutput)+1
            cpt_action=0
            while colonne_pleine(state, action):
                cpt_action+=1
                action=numpy.argsort(-myNN_past.LastLayerOutput)[cpt_action]+1
                    
            
            next_state=jouer_bis(state, joueur_actuel, action)
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie_bis(state, joueur_actuel, cpt_victoire)
        
        
            
        if (next_cpt_victoire[0]-cpt_victoire[0])==1:
            reward=1
        elif (next_cpt_victoire[1]-cpt_victoire[1])==1:
            reward=-1
        else :
            reward=0
        x=state
            
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)

        target = reward+gamma*next_max
   
        x=state
        myNN.retropropagation(x,target,action-1)
  
        state = next_state
        joueur_actuel=joueur_suivant    
    
        cpt_victoire=next_cpt_victoire
        nbSteps=nbSteps+1
        
print("end of learning period")


#%% test myNN vscontre ia random

nbEpisodes=1000
cpt_vic=[0,0,0]
next_cpt_vic=[0,0,0]
for i in range(0, nbEpisodes):
    if i%50==0:
        print("episode: "+str(i))
        successesInARow=0

    state=initialise_liste_positions()
    
    #--------------------
    tirage_joueur=random.randint(1,2)
    if tirage_joueur==1:
        joueur_actuel="yellow"
    else :
        joueur_actuel="red"
    #--------------------
    
    firstState=state
    endOfEpisode = False
    nbSteps=0

    while not endOfEpisode:
        if joueur_actuel=="yellow":
            x=state
            myNN.output(x)
            action = numpy.argmax(myNN.LastLayerOutput)+1
            #action = random.randint(1, 7)
            cpt_action=0
            while colonne_pleine(state, action):
                cpt_action+=1
                action=numpy.argsort(-myNN.LastLayerOutput)[cpt_action]+1

            next_state=jouer_bis(state, joueur_actuel, action)

            endOfEpisode, joueur_suivant, next_cpt_vic = fin_partie_bis(next_state, joueur_actuel, cpt_vic)
            
            x=state
            
            state = next_state
            joueur_actuel=joueur_suivant

        elif joueur_actuel=="red":
            action = random.randint(1, 7)
            while colonne_pleine(state, action):
                action = random.randint(1, 7)
            
            next_state=jouer_bis(state, joueur_actuel, action)
            endOfEpisode, joueur_suivant, next_cpt_vic = fin_partie_bis(next_state, joueur_actuel, cpt_vic)

            state = next_state
            joueur_actuel=joueur_suivant
            
        if (next_cpt_vic[0]-cpt_vic[0])==1:
            reward=1
        elif (next_cpt_vic[1]-cpt_vic[1])==1:
            reward=-1
        else :
            reward=0

        cpt_vic=next_cpt_vic
        
        nbSteps=nbSteps+1

print(cpt_vic)

#%% Joueur contrie myNN

print("Grille initial :")
cpt_vic=[0,0,0]
state=initialise_liste_positions()   
affiche_grille_console(state)   
firstState=state
endOfEpisode = False

#--------------------3
tirage_joueur=random.randint(1,2)
if tirage_joueur==1:
    joueur_actuel="yellow"
    print("Vous jouez en 1er :")
else :
    joueur_actuel="red"
    print("Vous jouez en 2eme, l'advsersaire joue : ")    
#--------------------


while not endOfEpisode:
    
    if joueur_actuel=="red":
       x=state
       myNN.output(x)
       action = numpy.argmax(myNN.LastLayerOutput)+1
       cpt_action=0
       while colonne_pleine(state, action):
            cpt_action+=1
            action=numpy.argsort(-myNN.LastLayerOutput)[cpt_action]+1
       next_state=jouer(state, joueur_actuel, action)
       endOfEpisode, joueur_suivant, next_cpt_vic = fin_partie(next_state, joueur_actuel, cpt_vic)
       state = next_state
       joueur_actuel=joueur_suivant
       
    elif joueur_actuel=="yellow":
       choix_col=input("choisir une colonne (1 à 7): ")
       action=int(choix_col)
       while colonne_pleine(state, action):
           choix_col=input("colonne pleine, choisir une colonne : ")
           action=int(choix_col)
       next_state=jouer(state, joueur_actuel, action)
       endOfEpisode, joueur_suivant, next_cpt_vic = fin_partie(next_state, joueur_actuel, cpt_vic)
       state = next_state
       joueur_actuel=joueur_suivant
            
    cpt_vic=next_cpt_vic