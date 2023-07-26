import java.awt.Color;

import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import javax.swing.JPanel;

public class Taquin extends JPanel { /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
//le damier sera mis dans un  Panel

// Taille du Taquin (nombre de ligne et nombre de colonne)
private int size;
// Nombre de tuiles
private int nbTiles;
// taille de la grille UI
private int dimension;
// couleur de fond
private static final Color FOREGROUND_COLOR = new Color(239, 83, 80); // couleur arbitraire
// objet Random pour melanger les tuiles
private static final Random RANDOM = new Random();
// Stock des tuiles dans n vecteur 1D sous forme d'entiers
// (case vide =0)
private int[] tiles;
// taille des tuiles UI
private int tileSize;
// position de la case vide
private int blankPos;
// marge de la grille UI
private int margin;
// taille grille UI 
private int gridSize;
// true si le jeux est fini, false sinon
private boolean gameOver;
private boolean gameSolve; // s'apparente au gameOver mais version Solveur
private int nb_etape=0; //compteur d'etape
//Liste des actions realiser
public ArrayList<Integer> History = new ArrayList<Integer>();
//liste des action annulee que l'on peut refaire
public ArrayList<Integer> Redo_List = new ArrayList<Integer>();

public Taquin(int size, int dim, int mar) {
 //parametres
 this.size = size;
 dimension = dim;
 margin = mar;
 
 // initialisation des tuiles 
 nbTiles = size * size - 1; // -1 car on ne compte pas la case vide
 tiles = new int[size * size];
 
 // calcule la taille de la grille et des tuiles
 gridSize = (dim - 2 * margin);
 tileSize = gridSize / size;
 
 gameOver = true;
 gameSolve= true;

 
 addMouseListener(new MouseAdapter() {
   @Override
   public void mousePressed(MouseEvent e) {
	 // permet l'interaction avec le damier en cliquant
	 
	   //si le jeux est resolu, relance un nouveau taquin
     if (gameOver || gameSolve) {
       newGame();
     } else {
       // position du clic
       int ex = e.getX() - margin;
       int ey = e.getY() - margin;
       
       // verification que le clic est dans la grille
       if (ex < 0 || ex > gridSize  || ey < 0  || ey > gridSize)
         return;
       
       // correspondance du clic dans la grille
       int c1 = ex / tileSize;
       int r1 = ey / tileSize;
       
       // position de la case vide
       int c2 = blankPos % size;
       int r2 = blankPos / size;
       
       // convertion en coordonnees 1D
       int clickPos = r1 * size + c1;
       int dir = 0;
       
       //on cherche la direction pour bouger plusieurs tuiles en meme temps
       if (c1 == c2  &&  Math.abs(r1 - r2) > 0)
         dir = (r1 - r2) > 0 ? size : -size;
       else if (r1 == r2 && Math.abs(c1 - c2) > 0)
         dir = (c1 - c2) > 0 ? 1 : -1;
         
       if (dir != 0) {
         // on deplace la tuile dans la direction
         do {
           int newBlankPos = blankPos + dir; //nouvelle position de la case vide
           tiles[blankPos] = tiles[newBlankPos]; //permutation
           History.add(blankPos); //en jouant un coup, on l'ajoute dans l'historique pour pouvoir l'annuler
           blankPos = newBlankPos; //mise a jour de la position de la case vide
         } while(blankPos != clickPos); // tant que la case vide n'est pas a sa nouvelle position
         
         tiles[blankPos] = 0; // mise a jour de la case vide
         Redo_List.clear(); // si on click, on ne peut plus refaire les actions annulees stockees
         
       }
       // on verifie si le jeux est fini
       gameOver = isSolved();
	   gameSolve= isSolved();
            
       
     }
     
     // mise a jour de l'affichage
     repaint();
  
    		
   }
 });
 
 
 //on debute un jeux
 newGame();
}

public void Undo_Last_Action() {
	// la fonction permet d'annuler la derniere action faite
	// peut etre utiliser successivement
	
	//on regarde le nombre d'action faites
	int size_list=History.size();	
	// si il y a une action que l'on peut annuler, on peut annuler la derniere faite :
	if (size_list>0) {
		do {
	           int newBlankPos = History.get(size_list-1); //recherche de l'action a annuler
	           tiles[blankPos] = tiles[newBlankPos]; //permutation de la tuile
	           Redo_List.add(blankPos); //on ajoute l'action dans la liste des action que l'on peut refaire
	           blankPos = newBlankPos; //nouelle position de la case vide
	         } while(blankPos != History.get(size_list-1));
		History.remove(size_list-1); //l'action annuler ne fait plus partie des actions faites
		tiles[blankPos] = 0; //mise a jour de la case vide
	}    
	repaint(); //mise a jour graphique

}

public void Redo_Last_Action() {
	// la fonction permet de refaire la derniere action annulee avec undo
	// peut etre utiliser successivement
	
	//on regarde le nombre d'action annulee que l'on peut refaire
	int size_list=Redo_List.size();	
	// si il y a une action que l'on peut refaire, on peut refaire la derniare annulee :
	if (size_list>0) {
		do {
	           int newBlankPos = Redo_List.get(size_list-1); //recherche de l'action a faire
	           tiles[blankPos] = tiles[newBlankPos]; //permutation de la tuile
	           History.add(blankPos); // on remet l'action dans History car elle a etait realisa
	           blankPos = newBlankPos; //nouvelle position de la case vide
	         } while(blankPos != Redo_List.get(size_list-1));
		Redo_List.remove(size_list-1); //l'action annulee ne peut plus etre annulee
		tiles[blankPos] = 0; //mise a jour de la case vide
	}    
	repaint(); //mise a jour graphique
}


public void newGame() {
 do {
   reset(); // remet le jeux a l'etet initial
   shuffle(); // melange des tuiles
   History.clear(); //on vide l'historique des action
   Redo_List.clear(); //on vide les action que l'on peut refaire
 } while(!isSolvable() || isSolved()); // on refait ces actions jusqu'a avoir un damier solvable et non resolu
 
 gameOver = false;//le jeux n'est pas fini au debut
 gameSolve= false;
}

public void reset() {
	//on positionne les tuiles de 1 a size*size-1 sur les cases 0 a size*size-2
 for (int i = 0; i < tiles.length; i++) {
   tiles[i] = (i + 1) % tiles.length;
 }
 
 // on met la case vide a la fin
 blankPos = tiles.length - 1;
}

public void shuffle() {
 // la case vide n'est pas incluse
 int n = nbTiles;
 //pour chaque tuiles, on veut faire une permutation
 while (n > 1) {
   int r = RANDOM.nextInt(n--); //choix de permutation
 //permutation de tuile
   int tmp = tiles[r];
   tiles[r] = tiles[n]; 
   tiles[n] = tmp;
 }
}



public boolean isSolvable() {
	// Seulement la moitie des permtations sont solvable, la fonction verifie
	// que le damier actuelle est solvable
	
	//itialisation, stockage
	  boolean again = true ;
	  int countInversions = 0;
	  int [] num_case;
	  int permut ;
	  num_case=copie_list(tiles); 
	  
	  
	  while (again) {
	    again = false ;
	    for (int i = 0; i < nbTiles; i++) { //pour chaque tuile
	      if (num_case[i]!=i+1) { //si une tuile n'est pas a sa place
	        again = true ; //on continue a regarder
	        countInversions++ ;	 //on augmente le nombre de permutation
	        
	        //on fait la permuatation
	        permut = num_case[num_case[i]-1];
	        num_case[num_case[i]-1] = num_case[i];
	        num_case[i] = permut ;
	      }
	    }
	  }
	  //si le nombre de permutation necessire est paire, le damier est solvable
	  return countInversions % 2 == 0;
	}

public boolean isSolved() {
 if (tiles[tiles.length - 1] != 0) // si la case vide n'est pas a la fin, le taquin n'est pas resolu
   return false;
 
 for (int i = nbTiles - 1; i >= 0; i--) { // on verifie la position de chaque tuile
   if (tiles[i] != i + 1)
     return false;      
 }
 
 return true;
}


public void drawGrid(Graphics2D g) {
    for (int i = 0; i < tiles.length; i++) {
      // on transforme les coordonnees 1D en 2D en fonction de la taille du vecteur
      int r = i / size;
      int c = i % size;
      // on converti en coordonnee dans UI
      int x = margin + c * tileSize;
      int y = margin + r * tileSize;
      
      // on verifie la position de la case vide
      if(tiles[i] == 0) {
        if (gameOver || gameSolve) {
          g.setColor(FOREGROUND_COLOR);
          drawCenteredString(g, "\u2713", x, y);
        }
        
        continue;
      }
      
      // pour les autres tuiles
      // on les affiche en bleu, avec un numero blanc
      // et une frontiere noir
      g.setColor(Color.BLUE);
      g.fillRoundRect(x, y, tileSize, tileSize, 25, 25);
      g.setColor(Color.BLACK);
      g.drawRoundRect(x, y, tileSize, tileSize, 25, 25);
      g.setColor(Color.WHITE);
      
      drawCenteredString(g, String.valueOf(tiles[i]), x , y);
    }
  }
  
  public void drawStartMessage(Graphics2D g) {
	  // a la fin de la partie, on affiche un message
	  // pour commencer une nouvelle partie
    if (gameOver) {
	  gameSolve=false;
      g.setFont(getFont().deriveFont(Font.BOLD, 18));
      g.setColor(FOREGROUND_COLOR);
      String s = "Click to start new game";
      g.drawString(s, (getWidth() - g.getFontMetrics().stringWidth(s)) / 2,
          getHeight() - margin);
    }
  }

  public void drawStartMessageSolve(Graphics2D g) {
	  // a la fin du solve, on affiche un message
	  // pour commencer une nouvelle partie ainsi que
	  // le nombre d'etapes du solveur
	    if (gameSolve) {
	      g.setFont(getFont().deriveFont(Font.BOLD, 18));
	      g.setColor(FOREGROUND_COLOR);
	      String s = " etapes pour resoudre le jeu. Click to restart.";
	      g.drawString(nb_etape+s, (getWidth() - g.getFontMetrics().stringWidth(s))/2,
	          getHeight() - margin);
	    }
	  }
  
  public void drawCenteredString(Graphics2D g, String s, int x, int y) {
    // on centre la string s pour la tuile en (x,y)
    FontMetrics fm = g.getFontMetrics();
    int asc = fm.getAscent();
    int desc = fm.getDescent();
    g.drawString(s,  x + (tileSize - fm.stringWidth(s)) / 2, 
        y + (asc + (tileSize - (asc + desc)) / 2));
  }
  
  @Override
  public void paintComponent(Graphics g) {
    super.paintComponent(g);
    Graphics2D g2D = (Graphics2D) g;
    g2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
    drawGrid(g2D);
    drawStartMessage(g2D);
	drawStartMessageSolve(g2D);
  }
  
 
 public int[] copie_list(int[] list)
 //la fonction copie la liste list pour
 //eviter de modifier la liste d'origine
 {
	 //stocakge
	 int dim_list=size*size;
	 int[] new_list= new int[dim_list];
	 
	 //copie
	 for(int i=0; i<dim_list;i++) {
		 int new_elem=list[i];
		 new_list[i]=new_elem;
	 }
	 
	 return new_list;
 }
  
 public ArrayList<int[]> Successeur (int[] tuiles){
	 // la fonction permet de voir les successeur possible d'un etat
	 
	 //stockage et initialisation
	 int nbr_ligne=size;
	 ArrayList<int[]> Succes=new ArrayList<int[]>();
	 int position_vide=0;
	 
	 //on cherceh la case vide
	 for (int i=0; i<size*size;i++) {
		 if (tuiles[i]==0) {
			 position_vide=i;
		 }
	 }
	 
	 //si il y a une tuile en haut de la case vide, on permute pour avoir un successeur
	 if (position_vide-nbr_ligne >= 0) {
		 //stockage
		 int[] haut;
		 //copie de la liste d'origine pour ne pas la modifier
		 haut=copie_list(tuiles);
		 //permutation
		 haut[position_vide]=tuiles[position_vide-nbr_ligne];
		 haut[position_vide-nbr_ligne]=tuiles[position_vide]; 
		 //stockage
		 Succes.add(haut);
	 }
	//si il y a une tuile en bas de la case vide, on permute pour avoir un successeur
	 if (position_vide+nbr_ligne < size*size) {
		//stockage
		 int[] bas;
		//copie de la liste d'origine pour ne pas la modifier
		 bas=copie_list(tuiles);
		//permutation
		 bas[position_vide]=tuiles[position_vide+nbr_ligne];
		 bas[position_vide+nbr_ligne]=tuiles[position_vide]; 
		//stockage
		 Succes.add(bas);
	 }
	//si il y a une tuile a droite de la case vide, on permute pour avoir un successeur
	 if ((position_vide+1)%size != 0) {
		//stockage
		 int[] droite;
		//copie de la liste d'origine pour ne pas la modifier
		 droite=copie_list(tuiles);
		//permutation
		 droite[position_vide]=tuiles[position_vide+1];
		 droite[position_vide+1]=tuiles[position_vide];
		//stockage
		 Succes.add(droite);
	 }
	//si il y a une tuile a gauche de la case vide, on permute pour avoir un successeur
	 if ((position_vide)%size != 0) {
		//stockage
		 int[] gauche;
		//copie de la liste d'origine pour ne pas la modifier
		 gauche=copie_list(tuiles);
		//permutation
		 gauche[position_vide]=tuiles[position_vide-1];
		 gauche[position_vide-1]=tuiles[position_vide];
		//stockage
		 Succes.add(gauche);
	 }
	 return Succes;
	 
 }
 
public int distance_manhattan(int[] tuiles) {
	// calcul la distance de manhattan d'un damier a sa resolution
	// c'est a dire, la somme pour chaque tuiles de 
	// la distance en ligne entre une tuile et sa position resolu
	// et de la distance en colonne entre une tuile et sa position resolu
	
	//stockage
	int[] copie_tiles=copie_list(tuiles);
	int dist=0;
	int ligne=0;
	int colonne=0;
	int t_col=0;
	int t_lig=0;
	int d_col=0;
	int d_lig=0;
	int t=0;
	
	//pour chaque tuile :
	for (int i=0; i<size*size; i++) {
		//position de la case actuelle
		colonne=i%size;
		ligne=(i-colonne)/size;
		
		//tuile dans la case actuelle
		t=copie_tiles[i];
		if (t==0) {
			// si on a une case vide, on fixe la tuile a size*size
			// pour avoir le meme macanisme que les autres tuiles
			t=size*size; 
		}
		//position resolu de la tuile observee
		t_col=(t-1)%size;
		t_lig=(t-1-t_col)/size;
		
		//distance a sa position resolu
		d_col=Math.abs(colonne-t_col);
		d_lig=Math.abs(ligne-t_lig);
		
		//mise a jour distance de manhattan
		dist=dist+d_col+d_lig;
	}
	return dist;
}

public static int argMin(ArrayList<Integer> a) {
	// retourne la position de la valeur minimum dans une liste
	
	//initialisation
    int v = Integer.MAX_VALUE;
    int ind = -1;
    
    //pour chaque element de la liste
    for (int i = 0; i < a.size(); i++) {
        if (a.get(i) < v) { // mise a jour du minimum et de sa position
            v = a.get(i);
            ind = i;
        }
    }
    return ind;
}
  
public boolean dans_liste(ArrayList<int[]> liste ,int[] element) {
	// verifie qu'un element est dans une liste
	
	//initialisation
	boolean again=true;
	int[] test;
	int cpt=0;
	
	//on parcours la liste
	for (int i=0;i<liste.size();i++) {
		cpt=0;
		test=liste.get(i);
		again=true;
		
		//on verifie si l'alament de la liste actuelle correspond a l'element
		while((again) && (cpt<element.length)) {
			if (test[cpt]!=element[cpt]) { //si un composant ne corrrespond pas
				again=false; //plus besoin de continuer
			}
			else {
				cpt++;//on continue avec la verification suivante
			}
		}
		//si tout correspond, on return true
		if (cpt==element.length){
			return true;
		}
	}
	return false;
}


public int pos_dans_liste(ArrayList<int[]> liste ,int[] element) {
	//retourne la position d'un element dans une liste
	//il faut donc verifier s'il est dans la liste
	
	//initialisation
	boolean again=true;
	int[] test;
	int cpt=0;
	
	//on parcours la liste
	for (int i=0;i<liste.size();i++) {
		cpt=0;
		test=liste.get(i);
		again=true;
		while((again) && (cpt<element.length)) {
			if (test[cpt]!=element[cpt]) {//si un composant ne corrrespond pas
				again=false;//plus besoin de continuer
			}
			else {
				cpt++;//on continue avec la verification suivante
			}
		}
		//si tout correspond, on retourne la position dans la liste
		if (cpt==element.length){
			return i;
		}
	}
	//si il n'y a pas de correspondance, on retourne -1
	return -1;
}

 public void Solveur_Taquin()  {
	 //permet de resoudre le Taquin actuelle etape par etape
	 
	 //stockage
	 ArrayList<int[]> ouverte = new ArrayList<int[]>(); //damier disponibles
	 ArrayList<int[]> fermee = new ArrayList<int[]>(); //damier utilisas
	 ArrayList<Integer> fn = new ArrayList<Integer>(); //les couts
	 ArrayList<int[]> temp=new ArrayList<int[]>(); //stock temporairement les successeur consideres
	 ArrayList<int[]> stock_parent=new ArrayList<int[]>();//les damiers obtenus
	 ArrayList<Integer> indice_parent=new ArrayList<Integer>();//la position du damier parents dans stock_parents
	 ArrayList<Integer> chemin = new ArrayList<Integer>(); //indices des damiers pour avoir le chemin de resolution
	 int[] temp_actu; //stock un successeur a la fois
	 ArrayList<Integer> nb_coups = new ArrayList<Integer>(); //le nombre de coup pour obtenir les damiers
	 
	 
	 
	 //initialisation
	 int num_parent=0; 
	 int score=0;
	 nb_coups.add(0); // le damier initiale est obtenue en 0 coup
	 Integer a=distance_manhattan(tiles); //distance initiale
	 fn.add(a); //distance initiale
	 boolean trouve=false;
	 boolean sans_sol=false;
	 int[] copie_tiles=copie_list(tiles);
	 int[] dam_actu=copie_list(tiles);
	 int arg=0;
	 ouverte.add(copie_tiles); //on stock le damier initiale dans les damier observable
	 stock_parent.add(copie_tiles); //on stock le damier initiale dans stock_parent
	 indice_parent.add(-1); //sans parents, -1 de facon arbitraire
	 
	 //tant que l'on ne trouve pas de solution
	 while (!trouve) {
		 //si il n'y a plus de damier utilisable on s'arrate
		 if (ouverte.size()==0) {
			 trouve=true;
			 System.out.println("pas de solution trouvae");
			 sans_sol=true;
		 }
		 
		 
		 else {
			 arg=argMin(fn); // position du damier au cout initiale
			 dam_actu=ouverte.get(arg); // selection du damier
			 temp=Successeur(dam_actu); //successeur du damier
			 fermee.add(dam_actu); // on ajoute le damier au groupe daja utilisas
			 
			 //pour chaque successeur
			 for(int i=0; i<temp.size(); i++) {
				 temp_actu=temp.get(i);
				 
				 //on calcul le cout en additionnant la distance de manhattan et le nombre de coup necessaire
				 // pour avoir le succeseur
				 
				 //dimension 2 et 3 marche
				 //dimension 4 est resolu plus rapidment en changeant la fonction score
				 // dimenson 5 est trop long pour etre execute
				 if (size==4) {
					 score=100*distance_manhattan(temp_actu)+nb_coups.get(pos_dans_liste(ouverte,dam_actu))+1;
				 }
				 else {
					 score=distance_manhattan(temp_actu)+nb_coups.get(pos_dans_liste(ouverte,dam_actu))+1;
				 }
				
				 // si il est deja parmis les damiers observes mais a un meilleur score, on met a jour
				 if (dans_liste(ouverte,temp_actu)) {
					 if (fn.get(pos_dans_liste(ouverte,temp_actu))>score) {
						 fn.set(pos_dans_liste(ouverte,temp_actu),score); //le score est mis a jour
						 nb_coups.set(pos_dans_liste(ouverte,temp_actu),nb_coups.get(pos_dans_liste(ouverte,dam_actu))+1); //le nombre de coups est mis a jour
						 indice_parent.set(pos_dans_liste(stock_parent,temp_actu),pos_dans_liste(stock_parent,dam_actu)); // le parent est mis a jour
					 }
				 }	 
				 
				 //si il s'agit d'un nouveau cas, on l'ajoute et on calcul son score
				 else if(!dans_liste(fermee,temp_actu)) {
					ouverte.add(temp_actu); //on ajoute le successeur au damier a observer
					indice_parent.add(pos_dans_liste(stock_parent,dam_actu)); //on definit son parent
					stock_parent.add(temp_actu); //on ajoute le successeur aux observations
					fn.add(score);//on ajoute son score
					nb_coups.add(nb_coups.get(pos_dans_liste(ouverte,dam_actu))+1);//coup pour le parent +1
				 }
			 }
			 
			//on enleve le damier actuelle (parent des successeurs)
			// des damiers a utiliser, ainsi que ces informations associees
			fn.remove(pos_dans_liste(ouverte,dam_actu));
			nb_coups.remove(pos_dans_liste(ouverte,dam_actu));
			ouverte.remove(pos_dans_liste(ouverte,dam_actu));
			
			// on verifie si le damier actuelle (parent des successeurs)
			// est resolu
			if (distance_manhattan(dam_actu)==0) {
				trouve=true;
				
			}		
			
	 }
	 
	 
	 
	 }
	 //si un damier est resolu, on reconstruit le chemin
	 if (trouve==true && sans_sol==false) {
		 //initialisation du chemin :
		 num_parent=pos_dans_liste(stock_parent,dam_actu); //position de son parent
		 chemin.add(num_parent);
		 
		 //tant que l'on ne retrouve pas le damier initial, on continue le chemin
		 while(num_parent!=0) {
			 //on ajoute le parents du dernier damier du chemin
			 num_parent=indice_parent.get(num_parent);
			 chemin.add(num_parent);
		 }
		 
		 //on remonte le chemin pour resoudre etape par etape
		 for (int j=chemin.size()-1;j>-1;j--) {
			//possibilite de faire le chemin inverse avec undo
			 History.add(blankPos);
			 
			//copie de la position suivante
			 int[] new_tiles=stock_parent.get(chemin.get(j));
			 System.out.println("ETAPE "+(chemin.size()-j)+" ---------");
			 
			 //initialisation
			 String damier="";
			 int cpt_affichage=0;
			 int case_vide=0;
			 
			 //position case vide
			 for (int y=0;y<size*size;y++) {
				 if (new_tiles[y]==0) {
					 case_vide=y;
				 }
			blankPos=case_vide;
				 
				 
			 }
			 //pour chaque tuile on met a jour
			 for (int k=0;k<size*size;k++) {
				 tiles[k]=new_tiles[k]; //remplacement
				 
				 
				//construction sortie console
				 damier+=new_tiles[k];
				 cpt_affichage++;
				 damier+=" ";
				 if (cpt_affichage==size) {
					 System.out.println(damier);
					 damier="";
					 cpt_affichage=0;
				 }
			 }
			 repaint(); //mise a jour graphique
			 
			 //pause entre chaque mouvement
			 //nous n'avons pas reussi a le faire
			 //Thread.sleep(500);
			 //TimeUnit.SECONDS.sleep(1);
			 
		 
		 }
		 
	 }
	 

	gameSolve=true;
	nb_etape=chemin.size();
}
}