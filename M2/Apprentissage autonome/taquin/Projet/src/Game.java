import java.awt.Dimension;
import javax.swing.JFrame;
import javax.swing.*;
import java.awt.Rectangle;  
import javax.swing.JButton;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class Game extends JFrame implements ActionListener{
	//les boutons :
	private JButton btnRestart;
	private JButton btnSolve;
	private JButton btnUndo;
	private JButton btnRedo;
	private JComboBox<String> comboBoxDimension;
	
	//jeux
	private Taquin taquin;
	
	//fenetre
	private JFrame frame;
	
	//parametre
	private int dim_actuelle;

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public Game(int dim) {
		//on definit la fenetre
	   	  frame = new JFrame();
		  frame.setBounds(new Rectangle(0, 0, 800, 530)); // taille image
          JLabel l1 = new JLabel("Pour resoudre le taquin, il faut remettre les");
          l1.setBounds(500,0,449, 471); 
          frame.add(l1);
          JLabel l2 = new JLabel("tuiles dans l'odre en cliquant dessus.");
          l2.setBounds(500, 20, 449, 471);
          frame.add(l2);
	      frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); //on ferme le programme en fermant la fenetre
	      frame.setTitle("Jeux du Taquin"); //titre
	      frame.setResizable(false); //non redimensionnable
	      frame.getContentPane().setLayout(null); 
	      taquin = new Taquin(dim, 450, 30); //on debute le jeux
	      taquin.setBounds(40, 0, 449, 471); //dimension du taquin
	      frame.getContentPane().add(taquin); //on met le taquin sur la fenetre
	      
	      JMenuBar menuBar = new JMenuBar(); //barre de menu
	      menuBar.setPreferredSize(new Dimension(100, 30)); // dimension du menu
	      frame.setJMenuBar(menuBar); // on attache le meu a la fenetre
	      
	      //bouton restart
	      btnRestart = new JButton("Restart"); 
	      btnRestart.addActionListener(this);
	      menuBar.add(btnRestart);
	      
	      //bouton solve
	      btnSolve = new JButton("Solve");
	      btnSolve.addActionListener(this);
	      menuBar.add(btnSolve);
	      
	      //bouton redo
	      btnRedo = new JButton("Redo");
	      btnRedo.addActionListener(this);
	      menuBar.add(btnRedo);
	      
	      //bouton undo
	      btnUndo = new JButton("Undo");
	      btnUndo.addActionListener(this);
	      menuBar.add(btnUndo);
	      
	      //bouton choix de dimension
	      comboBoxDimension = new JComboBox<String>();
	      comboBoxDimension.setModel(new DefaultComboBoxModel<String>(new String[] {"Dimension 2x2","Dimension 3x3", "Dimension 4x4", "Dimension 5x5"}));
	      comboBoxDimension.setSelectedIndex(dim-2);
	      comboBoxDimension.addActionListener(this);
	      menuBar.add(comboBoxDimension);
	      
	      //on affiche la fenetre
	      pack();
	      frame.setVisible(true);
	      
	      //parametre
	      dim_actuelle=dim;
	      //-----------------------------------
	      
	      

	}

	@Override
	public void actionPerformed(ActionEvent btn) {
		// TODO Auto-generated method stub
		
		//on recupere le bouton utilise
		Object source = btn.getSource();
		 
		if(source == btnRestart){
			//on recommence le jeux
			frame.dispose();
			new Game(dim_actuelle);
		}
		else if(source == btnSolve){
			//on resout le taquin
			//etape affichees en console
			//optimisee pour dimension 2 et 3
			//fonctionne pour dimension 4
			// probleme de temps sur dimension 5
			taquin.Solveur_Taquin();
			
		}
		else if(source == btnRedo){
			//refaire la derni�re action undo
			//peut etre realiser successivement
			taquin.Redo_Last_Action();	
		}
		else if(source == btnUndo){
			//enleve la derniere action realise
			//peut etre realiser successivement
			taquin.Undo_Last_Action();	
		}
		else if(source == comboBoxDimension) {
			//on recupere la dimension choisit
			String stock_txt=comboBoxDimension.getSelectedItem().toString();
			
			//on relance avec la nouvelle dimension
			if (stock_txt=="Dimension 2x2") {
				int dim_choisit=2;
				frame.dispose();
				new Game(dim_choisit);
			}
			else if (stock_txt=="Dimension 3x3") {
				int dim_choisit=3;
				frame.dispose();
				new Game(dim_choisit);
			}
			else if (stock_txt=="Dimension 4x4") {
				int dim_choisit=4;
				frame.dispose();
				new Game(dim_choisit);
			}
			else if (stock_txt=="Dimension 5x5") {
				int dim_choisit=5;
				frame.dispose();
				new Game(dim_choisit);
			}

		}
	}
}
