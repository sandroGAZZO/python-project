# PROJET PUISSANCE 4

import random
import numpy
import math

#%%

# REGLAGES POUR LA COMPETITION
NB_PARTIES = 1000000  # A adapter (un nombre pair plus grand que 1)


# CONTRAINTES DU JEU
NB_COLONNES = 7  # Nombre de colonnes de la grille de jeu
NB_LIGNES = 6  # Nombre de lignes de la grille de jeu
ALIGNEMENT = 4  # Nombre de pions à aligner pour gagner

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
# AFFICHAGE DES MESSAGES DANS LA CONSOLE
########################################################################

def affiche_joueur_console(couleur):
    """Affichage du joueur dans la console"""
    if couleur == 'yellow':
        print('Les jaunes jouent')
    elif couleur == 'red':
        print('Les rouges jouent')

def affiche_gagnant_console(couleur):
    """Affichage du gagnant dans la console"""
    if couleur == 'yellow':
        print('Les jaunes gagnent', end='')
    elif couleur == 'red':
        print('Les rouges gagnent', end='')

def affiche_aucun_gagnant_console():
    """Affichage aucun gagnant dans la console"""
    print('Aucun gagnant')

#%%

########################################################################
# MOTEUR DU JEU
########################################################################

def initialise_liste_positions():
    """Vide la grille"""
    return [0] * NB_COLONNES*NB_LIGNES

def alignement(somme, nbPions, couleur):
    """Analyse la somme dont il est question dans alignements_pleins() ou alignements_troues() pour détermminer si des pions sont alignés"""
    pionsAlignes = False
    if (couleur == 'yellow' and somme == nbPions) or (couleur == 'red' and somme == -nbPions):
        pionsAlignes = True
    return pionsAlignes

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

def grille_pleine(positions):
    """Teste si la grille est pleine"""
    plein = True
    for i in range(NB_LIGNES*NB_COLONNES):
        if positions[i] == 0:
            plein = False
    return plein

def inverse(couleur):
    """ Inverse les couleurs"""
    if couleur == 'yellow':
        couleur = 'red'
    elif couleur == 'red':
        couleur = 'yellow'
    return couleur


def colonne_pleine(positions, colonne):
    """Teste si la colonne indiquée est pleine"""
    plein = True
    position = NB_COLONNES*(NB_LIGNES-1)+colonne-1
    if positions[position] == 0:
        plein = False
    return plein


   
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
    return positions

def fin_partie(positions, couleur, victoires):
    """ Test de fin de partie """
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


def symbole(state):
    """ Convertit la grille, en mettant les symbole X et O pour les pions"""

    n=len(state)
    sortie=[]
    
    for i in range (0,n):
        if state[i]==1:
            sortie.append("X")
        elif state[i]==-1:
            sortie.append("O")
        elif state[i]==0:
            sortie.append(" ")
    return sortie

#%% Création d'un adversaire


########################################################################
# STRATEGIES DE JEU
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

def liste_indices_maximum(liste):
    """Renvoie les indices des maximums d'une liste"""
    maxi = max(liste)
    indices = []
    for i in range(len(liste)):
        if liste[i] == maxi:
            indices += [i]
    return indices

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

def position_fonction(colonne, ligne):
    """Déduit d'une position dans la grille une position dans la liste positions[]"""
    position = (colonne-1) + (ligne-1)*NB_COLONNES
    return position

def colonne_extraite(position):
    """Déduit d'une position dans la grille la colonne correspondante"""
    colonne = position % NB_COLONNES + 1
    return colonne

def position_potentielle(positions, colonne, ligne):
    """Teste si une position est possible (case vide et support pour soutenir le pion)"""
    test = False
    if colonne >= 1 and colonne <= NB_COLONNES and ligne >= 1 and ligne <= NB_LIGNES:
        if positions[position_fonction(colonne, ligne)] == 0:  # Position libre
            test = True
            if ligne > 1:
                if positions[position_fonction(colonne, ligne - 1)] == 0:  # Ligne support inexistante
                    test = False
    return test

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
                        positionsPotentielles += [position_fonction(c + j, l)]
            if d == "V": # Vertical
                if position_potentielle(positions, c, l + nbPions):
                    positionsPotentielles += [position_fonction(c, l + nbPions)]
            if d == "DM": # Diagonale Montante
                for j in range(nbPions + 1):
                    if position_potentielle(positions, c + j, l + j):
                        positionsPotentielles += [position_fonction(c + j, l + j)]
            if d == "DD": # Diagonale Descendante
                for j in range(nbPions + 1):
                    if position_potentielle(positions, c + j, l - j):
                        positionsPotentielles += [position_fonction(c + j, l - j)]
    colonne = -1
    if len(positionsPotentielles) > 0:
        colonne = colonne_extraite(meilleure_position(positionsPotentielles))
    return colonne

def priorite_trouee(positions, nbPions, couleur):
    """Retourne une colonne où jouer"""
    listeAlignementsTroues = alignements_troues(positions, nbPions-1, couleur)
    return positions_potentielles(positions, listeAlignementsTroues)


########################################################################
# Définir une IA avec une stratégie de jeux 
########################################################################

def jouer_ordi_ia(positions, couleur):                             # Priorité Alignements Troués
    """IA joue"""
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
    elif colB2PH != -1: return jouer(positions, couleur, colB2PH)    # B2PH : L'IA essaye d'empêcher l'adversaire d'aligner 2 pions
    elif colA2PH != -1: return jouer(positions, couleur, colA2PH)
    else: return jouer_ordi_poids_cases(positions, couleur)# PH   : L'IA joue dans la case qui a le plus de poids




def reverse_list(state):
    """
    inverse le signe dans une liste, ici inverse les pions
    """
    n=len(state)
    state_inv=[]
    for i in range (0,n):
        state_inv.append(-state[i])
    return state_inv

#%% Réseau de neurones à 1 couche

#paramètre :
epsilon=0.15
gamma=0.8
alpha=0.4
alpha_reg=0.9999a

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

#création du réseau de neurones :
nbCells=NB_LIGNES*NB_COLONNES
sizeInput=nbCells
sizeHiddenLayer=100
sizeOutput=NB_COLONNES
myNN = NN(sizeInput, sizeHiddenLayer, sizeOutput)
cpt_victoire=[0,0,0]
next_cpt_victoire=[0,0,0]
action=1
next_state=initialise_liste_positions()

#%% apprentissage myNN vs ia.py
epsilon=0.1
gamma=0.9
alpha=0.5
alpha_reg=0.9999

nbEpisodes=20000

for i in range(0, nbEpisodes):
    alpha=alpha*alpha_reg #on diminue alpha a chaque partie
    
    #on affiche les 100 derniers résultats
    if i%100==0:
        print("episode: "+str(i))
        print("  -Score:"+str(cpt_victoire))
        cpt_victoire=[0,0,0]
    
    #initialisation
    state=initialise_liste_positions()

    #----tirage joueur -------
    tirage_joueur=random.randint(1,2)
    if tirage_joueur==1:
        joueur_actuel="yellow"
    else :
        joueur_actuel="red"
    #--------------------------

    #initialisation
    next_state=state
    endOfEpisode = False
    nbSteps=0
    
            
    while not endOfEpisode:
        if joueur_actuel=="yellow": #tor de myNN
            
            #exploration
            if random.uniform(0, 1) < epsilon:
                action = random.randint(1, NB_COLONNES)
                while colonne_pleine(state, action):
                    action = random.randint(1, NB_COLONNES)
                    
            else:
                #choix de l'action
                x=state
                myNN.output(x)
                action = numpy.argmax(myNN.LastLayerOutput)+1
                cpt_action=0
                test=0
                while colonne_pleine(state, action):
                    cpt_action+=1
                    action=numpy.argsort(-myNN.LastLayerOutput)[cpt_action]+1
            
            #mise a jur du jeux
            next_state=jouer(state, joueur_actuel, action)  
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie(next_state, joueur_actuel, cpt_victoire) 
            
            
            
        elif joueur_actuel=="red": #tour adverse
            #choix de l'action et mise a jour du jeux
            jouer_ordi_ia(state, "red")
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie(next_state, joueur_actuel, cpt_victoire)
        
        
        #choix des récompenses
        if (next_cpt_victoire[0]-cpt_victoire[0])==1: #victoire de myNN
            reward=1
        elif (next_cpt_victoire[1]-cpt_victoire[1])==1: #défaite de myNN
            reward=-1
        else : #égalité
            reward=0
    
        #qlearning, mise a jour de myNN
        x=state
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)
        target = reward+gamma*next_max
        x=state
        myNN.retropropagation(x,target,action-1)
        
        #mise a jour des informations
        state = next_state
        joueur_actuel=joueur_suivant    
        cpt_victoire=next_cpt_victoire
        nbSteps=nbSteps+1
        
print("end of learning period")

#%% apprentissage myNN vs ia rand
epsilon=0.1
gamma=0.9
alpha=0.5
alpha_reg=0.9999

nbEpisodes=20000

for i in range(0, nbEpisodes):
    alpha=alpha*alpha_reg #on diminue le alpha a chaque épisode
    
    #on affiche les scores des 100 derniers épisodes
    if i%100==0:
        print("episode: "+str(i))
        print("  -Score:"+str(cpt_victoire))
        cpt_victoire=[0,0,0]
        
    #initialisation
    state=initialise_liste_positions()
    
    #------ tirage joueur ----
    tirage_joueur=random.randint(1,2)
    if tirage_joueur==1:
        joueur_actuel="yellow"
    else :
        joueur_actuel="red"
    #--------------------------
    
    #initilisation
    next_state=state
    endOfEpisode = False
    nbSteps=0

    while not endOfEpisode:
        if joueur_actuel=="yellow": #tour de myNN
            
            #exploration
            if random.uniform(0, 1) < epsilon:
                action = random.randint(1, NB_COLONNES)
                while colonne_pleine(state, action):
                    action = random.randint(1, NB_COLONNES)
                    
            #choix de l'action        
            else:
                x=state
                myNN.output(x)
                action = numpy.argmax(myNN.LastLayerOutput)+1
                cpt_action=0
                while colonne_pleine(state, action):
                    cpt_action+=1
                    action=numpy.argsort(-myNN.LastLayerOutput)[cpt_action]+1
                    
            #mise a jour du jeux        
            next_state=jouer(state, joueur_actuel, action)       
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie(next_state, joueur_actuel, cpt_victoire) 
            
        elif joueur_actuel=="red": #tour adverse
            #choix de l'action aléatoire
            action = random.randint(1, NB_COLONNES)
            while colonne_pleine(state, action):
                action = random.randint(1, NB_COLONNES)
            
            #mise a jour du jeux
            next_state=jouer(state, joueur_actuel, action)
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie(next_state, joueur_actuel, cpt_victoire)
            
        #choix des récompenses
        if (next_cpt_victoire[0]-cpt_victoire[0])==1: #victoire de myNN
            reward=1
        elif (next_cpt_victoire[1]-cpt_victoire[1])==1: #défaite de myNN
            reward=-1
        else : #égalité
            reward=0
            
        #qlearning, mise a jour de myNN
        x=state
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)
        target = reward+gamma*next_max
        x=state
        myNN.retropropagation(x,target,action-1)
        
        #mise a jour des informations
        state = next_state
        joueur_actuel=joueur_suivant    
        cpt_victoire=next_cpt_victoire
        nbSteps=nbSteps+1
        
print("end of learning period")

#%% apprentissange myNN vs myNN mise à jour de l'adversaire 
# toutes les 1OOO ittérations
epsilon=0.1
gamma=0.9
alpha=0.5
alpha_reg=0.9999

#parametre d'apprentissage
myNN_past=myNN
nbEpisodes=20000

for i in range(0, nbEpisodes):
    #mise a jour de myNN adverse
    if i%1000==0:
        myNN_past=myNN
    
    alpha=alpha*alpha_reg #diminution du lpha a chaque partie
    
    #on affiche les résultats des 100 dernières parties
    if i%100==0:
        print("episode: "+str(i))
        print("  -Score:"+str(cpt_victoire))
        cpt_victoire=[0,0,0]
    
    #initialisation
    state=initialise_liste_positions()
    
    #------- tirage joueur ---
    tirage_joueur=random.randint(1,2)
    if tirage_joueur==1:
        joueur_actuel="yellow"
    else :
        joueur_actuel="red"
    #-------------------------

    #initialisation
    next_state=state
    endOfEpisode = False
    nbSteps=0

    while not endOfEpisode:
        if joueur_actuel=="yellow": #tour de myNN
            
            #exploration
            if random.uniform(0, 1) < epsilon:
                action = random.randint(1, NB_COLONNES)
                while colonne_pleine(state, action):
                    action = random.randint(1, NB_COLONNES)
                    
            #choix de l'action
            else:
                x=state
                myNN.output(x)
                action = numpy.argmax(myNN.LastLayerOutput)+1
                cpt_action=0
                while colonne_pleine(state, action):
                    cpt_action+=1
                    action=numpy.argsort(-myNN.LastLayerOutput)[cpt_action]+1
                    
            #mise a jour du jeux
            next_state=jouer(state, joueur_actuel, action)       
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie(next_state, joueur_actuel, cpt_victoire) 
            
        elif joueur_actuel=="red": #myNN adverse
            #choix de l'action
            y=reverse_list(state)
            myNN_past.output(y)
            action = numpy.argmax(myNN_past.LastLayerOutput)+1
            cpt_action=0
            while colonne_pleine(state, action):
                cpt_action+=1
                action=numpy.argsort(-myNN_past.LastLayerOutput)[cpt_action]+1
                    
            #mise a jour du jeux
            next_state=jouer(state, joueur_actuel, action)
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie(next_state, joueur_actuel, cpt_victoire)
        
        
        #choix des récompenses
        if (next_cpt_victoire[0]-cpt_victoire[0])==1: #victoire de myNN
            reward=1
        elif (next_cpt_victoire[1]-cpt_victoire[1])==1: #défaite de myNN
            reward=-1
        else : #égalité
            reward=0
        
        #Qleraning, mise à jour du resau de neurones
        x=state
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)
        target = reward+gamma*next_max
        x=state
        myNN.retropropagation(x,target,action-1)
        
        #mise à jour des informations
        state = next_state
        joueur_actuel=joueur_suivant    
        cpt_victoire=next_cpt_victoire
        nbSteps=nbSteps+1
        
print("end of learning period")


#%% test myNN vscontre ia random

#initialisation des parametre et du compteur
nbEpisodes=1000
cpt_vic=[0,0,0]
next_cpt_vic=[0,0,0]

for i in range(0, nbEpisodes):
    
    #repère d'excution du code
    if i%50==0:
        print("episode: "+str(i))
        successesInARow=0
    
    #initialistion
    state=initialise_liste_positions()
    
    #---- Tirage joueur -----
    tirage_joueur=random.randint(1,2)
    if tirage_joueur==1:
        joueur_actuel="yellow"
    else :
        joueur_actuel="red"
    #-------------------------
    
    #initialistion
    next_state=state
    endOfEpisode = False
    nbSteps=0

    while not endOfEpisode:
        if joueur_actuel=="yellow": #tour de myNN
            #choix de l'action
            x=state
            myNN.output(x)
            action = numpy.argmax(myNN.LastLayerOutput)+1
            cpt_action=0
            while colonne_pleine(state, action):
                cpt_action+=1
                action=numpy.argsort(-myNN.LastLayerOutput)[cpt_action]+1
            
            #mise à jour du jeux
            next_state=jouer(state, joueur_actuel, action)
            endOfEpisode, joueur_suivant, next_cpt_vic = fin_partie(next_state, joueur_actuel, cpt_vic)
            x=state
            state = next_state
            joueur_actuel=joueur_suivant

        elif joueur_actuel=="red": #tour adverse
            #choix aléatoire de l'action
            action = random.randint(1, NB_COLONNES)
            while colonne_pleine(state, action):
                action = random.randint(1, NB_COLONNES)
            
            #mise à jour du jeux
            next_state=jouer(state, joueur_actuel, action)
            endOfEpisode, joueur_suivant, next_cpt_vic = fin_partie(next_state, joueur_actuel, cpt_vic)
            state = next_state
            joueur_actuel=joueur_suivant
    
        
        #mise à jour des informations
        cpt_vic=next_cpt_vic
        nbSteps=nbSteps+1

#on affiche les résultats
print("contre une ia aléatoire après apprentissage")
print(cpt_vic)



#%% Joueur contre myNN (1 partie)

#------ Rappel représentation ------
print("-----Rappel-----")
print("    : case vide")
print("   X: pion adverse")
print("   O: pion joueur")
print("----------------")
print(" ")
#-----------------------------------

#---- initialisation du jeux -------
print("Grille initial :")
cpt_vic=[0,0,0]
state=initialise_liste_positions()   
affiche_grille_console(symbole(state))  
firstState=state
endOfEpisode = False
#-----------------------------------

#------ TIRAGE ROUGE/JAUNE ---------
tirage_joueur=random.randint(1,2) 
if tirage_joueur==1:
    joueur_actuel="yellow"
    print("Vous jouez en 2eme")
else :
    joueur_actuel="red"
    print("Vous jouez en 1er")
#-----------------------------------


while not endOfEpisode:    
        if joueur_actuel=="yellow": #tour de myNN
            print("Tour adverse :")
            #choix de l'action
            x=state
            myNN.output(x)
            action = numpy.argmax(myNN.LastLayerOutput)+1
            cpt_action=0
            while colonne_pleine(state, action):
                cpt_action+=1
                action=numpy.argsort(-myNN.LastLayerOutput)[cpt_action]+1
           
            #mise à jour du jeux
            next_state=jouer(state, joueur_actuel, action)
            endOfEpisode, joueur_suivant, next_cpt_vic = fin_partie(next_state, joueur_actuel, cpt_vic)
            state = next_state
            joueur_actuel=joueur_suivant
            
            #sortie console
            affiche_grille_console(symbole(state))  
            if endOfEpisode==True:
                print("Défaite")
       
        elif joueur_actuel=="red": #tour du joueur
        
            #choix de l'action
            choix_col=input("A votre tour: choisir une colonne (1 à "+str(NB_COLONNES)+"): ")
            action=int(choix_col)
            while (action<1) or (action>NB_COLONNES):
                choix_col=input("numéro de colonne incorect, choisir une colonne (1 à "+str(NB_COLONNES)+"): ")
                action=int(choix_col)
            while colonne_pleine(state, action):
                choix_col=input("colonne pleine, choisir une colonne (1 à "+str(NB_COLONNES)+"): ")
                action=int(choix_col)
                while (action<1) or (action>NB_COLONNES):
                    choix_col=input("numéro de colonne incorect, choisir une colonne (1 à "+str(NB_COLONNES)+"): ")
                    action=int(choix_col)
            
            #mise à jour du jeux
            next_state=jouer(state, joueur_actuel, action)
            endOfEpisode, joueur_suivant, next_cpt_vic = fin_partie(next_state, joueur_actuel, cpt_vic)
            state = next_state
            joueur_actuel=joueur_suivant
            
            #sortie console
            affiche_grille_console(symbole(state))  
            if endOfEpisode==True:
                print("Victoire")
                
