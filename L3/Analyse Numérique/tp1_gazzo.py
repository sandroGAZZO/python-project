"""
TP1
Cimetta Marvin 
Gazzo Sandro
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA

#%%
"""--------------------------------------------
 DiffDiv : calcule les différences divisées
 à partir des points de xdata et les valeurs
 correspondantes de ydata
//--------------------------------------------
"""
def DiffDiv(xdata,ydata):
    n=np.size(ydata)
    cfs=ydata.copy()
    for j in range(1,n):# fait jusqu'à n-1
#        cfs[j:n] = (cfs[j:n]-cfs[j-1:n-1])/(xdata[j:n]-xdata[0:n-j])
        cfs[j:] = (cfs[j:]-cfs[j-1:-1])/(xdata[j:]-xdata[:n-j]) 
    return cfs


"""--------------------------------------------
 HornerNewton : calcule les valeur du polynôme
 d'interpolation en les points [t(1),....,t(n)]=t
 à partir des coefficients calculés avec DiffDiv
 et les points dans xdata
 le résultat est retourné dans y=[y(1),...,y(n)]
//--------------------------------------------
"""
def HornerNewton(cfs,xdata,t):
    n=np.size(cfs)
    nt=np.size(t)
    Id = np.ones(nt)
    y=cfs[n-1]*Id;
    for i in np.arange(n-2,-1,-1):
        y=cfs[i]*Id+y*(t-xdata[i]*Id)
    return y

#%%
"""
Exercice 1
"""
def f1(x):
    return x**2+2*x

"""
La fonction PlotFunction permet de tracer une fonction f 
sur un intervalle [-a,a] et avec une précision donnée.
"""
def PlotFunction(f,a,prec):
    tabcisse=np.arange(-a,a+prec,prec)
    plt.plot(tabcisse,f(tabcisse))

"""
La fonction PlotInterpUnif permet de tracer pour une certaine précision 
le polynôme d'interpolation uniforme de degré au plus k sur un intervalle [-a,a].
"""
def PlotInterpUnif(f,a,k,prec):
    x=np.linspace(-a,a,k+1,endpoint=True)
    t=np.arange(-a,a+prec,prec)
    y=f(x)
    cfs=DiffDiv(x,y)
    g=HornerNewton(cfs,x,t)
    plt.plot(t,g)

"""
La commande écrite ci dessous nous montrera les interpolations 
d'une fonction de degré au plus 0 jusqu'à 11.
"""
plt.figure(1)
for i in range (0,11):
    PlotInterpUnif(f1,3,i,0.1)
    
"""
On remarque qu'à partir de k=2, il n'y a plus rien qui change. 
On va donc voir quelle est la marge d'erreur quand k=2 et quand k=10.
"""

"""
La fonction diff_f_pk retourne la différence entre une fonction f quelconque
et l'interpolation de degré k de cette fonction.
"""
def diff_f_pk(f,a,k,prec):
    t=np.arange(-a,a+prec,prec)
    x=np.linspace(-a,a,k+1,endpoint=True)
    y=f(x)
    cfs=DiffDiv(x,y)
    return (f(t)-HornerNewton(cfs,x,t))

diff2=diff_f_pk(f1,3,2,0.1)
diff10=diff_f_pk(f1,3,10,0.1)
r2=LA.norm(diff2,np.inf)
r10=LA.norm(diff10,np.inf)

"""
r2 et r10 sont du même ordre de grandeur qui est extrêmement petit (10**(-15)!). 
On a donc une erreur très faible et ce, dès le degré 2, ce qui est plutôt logique
car la fonction f1 est un polynôme de degré 2.
"un peu mon neveu !!!"
"""
#%%
"""
Exercice 2
"""
def f2(x):
    return np.sin(np.pi*x)

"""
La fonction PlotErrInterpUnif permet de tracer la courbe de l'erreur d'interpolation 
en fonction du degré.
"""
def PlotErrInterpUnif(f,a,deg,prec):
    list=[]
    for k in deg:
        x=np.linspace(-a,a,k+1,endpoint=True)
        t=np.arange(-a,a+prec,prec)
        y=f(x)
        cfs=DiffDiv(x,y) 
        gk=HornerNewton(cfs,x,t)
        list.append(LA.norm((abs(f(t)-gk)),np.inf))
    listlog=np.log(list)    
    degré=[] 
    for i in deg:
        degré.append(i)    
    plt.yscale("linear")
    plt.xlabel("Evolution de l'erreur d'interpolation en fonction du degré")
    plt.plot(degré,listlog)

plt.figure(2)
PlotFunction(f2,2,0.1)
for i in range (20,25):
    PlotInterpUnif(f2,2,i,0.1)

plt.figure(3)
PlotErrInterpUnif(f2,2,np.arange(50)+1,0.1)
"""
L'erreur sera de plus en plus grande quand le degré sera de plus en plus grand.
Ce résultat n'est pas intuitif.
Lorsqu'on doit augmenter les degrés, la courbe d'interpolation correspondante est moins qualitative
aux bords de la fonction.
"""

def f3(x):
    return 1/(1+(x**2))

plt.figure(4)
PlotFunction(f3,2,0.1)
for i in range (20,31):
    PlotInterpUnif(f3,2,i,0.1)

plt.figure(5)
PlotErrInterpUnif(f3,2,np.arange(50)+1,0.1)

plt.figure(6)
PlotFunction(f2,5,0.1)
for i in range (20,31):
    PlotInterpUnif(f2,5,i,0.1)

plt.figure(7)
PlotErrInterpUnif(f2,5,np.arange(50)+1,0.1)

plt.figure(8)
PlotFunction(f3,5,0.1)
for i in range (20,31):
    PlotInterpUnif(f3,5,i,0.1)

plt.figure(9)
PlotErrInterpUnif(f3,5,np.arange(50)+1,0.1)
"""
On a bien ce qu'on avait prédit de notre observation précédente (essentiellement sur 
les bords de la fonction).
Il doit y avoir un problème en certains points d'interpolation.
"""


#%%
"""
Exercice 3
"""
"""
La fonction tcheb nous permet d'avoir les n abscisses de Tchebychev. 
NB: Plus n sera grand, plus la précision sera grande.
"""
def tcheb(a,b,n):
    x=[]
    for i in range (0,n):
        y=np.cos(((2*i+1)*np.pi)/(2*n))
        x=x+[((a+b)/2)+((b-a)/2)*y]
    return x

"""
La fonction PlotInterpTcheb est une variante de la fonction PlotInterpUnif vu à l'exercice 1.
La variante est, qu'ici, il n'y a plus la variable prec (qu'il falait la plus petite possible)
mais la variable n et à contrario, cette fonction sera plus précise si n est grand.
"""
def PlotInterpTcheb(f,a,k,n):
    x=np.linspace(-a,a,k+1,endpoint=True)
    t=tcheb(-a,a,n)
    y=f(x)
    cfs=DiffDiv(x,y)
    g=HornerNewton(cfs,x,t)
    plt.plot(t,g)

def diff_f_pk_tcheb(f,a,k,n):
    t=tcheb(-a,a,n)
    x=np.linspace(-a,a,k+1,endpoint=True)
    y=f(x)
    cfs=DiffDiv(x,y)
    u=[]
    for i in t:
        u=u+[(f(i)-HornerNewton(cfs,x,i))]
    return u

"""
La variante est identique que pour la fonction PlotInterpTcheb.
"""
def PlotErrInterpTcheb(f,a,deg,n):
    list=[]
    for i in deg:
        y=diff_f_pk_tcheb(f,a,i,n)
        list=list+[LA.norm(y,np.inf)]
    listlog=np.log(list)    
    degré=[] 
    for i in deg:
        degré.append(i)    
    plt.yscale("linear")
    plt.xlabel("Evolution de l'erreur d'interpolation en fonction du degré")
    plt.plot(degré,listlog)

plt.figure(10)
PlotFunction(f3,2,0.1)
for i in range (20,31):
    PlotInterpTcheb(f3,2,i,50)
    
plt.figure(11)
PlotErrInterpTcheb(f3,2,np.arange(50)+1,100)

plt.figure(12)
PlotFunction(f3,5,0.1)
for i in range (20,31):
    PlotInterpTcheb(f3,5,i,50)
    
plt.figure(13)
PlotErrInterpTcheb(f3,5,np.arange(50)+1,100)
"""
Il y a une amélioration, c'est à dire une diminution de l'erreur, pour l'un des deux cas.
"""
