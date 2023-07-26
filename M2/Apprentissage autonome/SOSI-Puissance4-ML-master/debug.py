#!/usr/bin/python
import puissance4
import afficheur
import joueur

p = puissance4.plateau()
a = afficheur.afficheurConsole(p)

print(p.alignementHorizontal(6,1))
a.afficherPlateau()

# j1 = joueur.Joueur(puissance4.pion.ROUGE,p)
# j2 = joueur.Joueur(puissance4.pion.JAUNE,p)
# for k in range(15):
#     j1.jouer()
#     a.afficherPlateau()
#     j2.jouer()
#     a.afficherPlateau()

for i in range(15):
    c = input()
    p.lacherPion(int(c), puissance4.pion.JAUNE)
    a.afficherPlateau()
    print(p.estFini())
    c = input()
    p.lacherPion(int(c), puissance4.pion.ROUGE)
    a.afficherPlateau()

    print(p.estFini())
