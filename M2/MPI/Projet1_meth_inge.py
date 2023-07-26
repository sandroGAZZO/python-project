import numpy as np
import matplotlib.pyplot as plt


#%%

h=1.0
L=10.0
x=np.arange(-L,L+1,h )
y=np.arange(-1,2.5,h/2)
x_m=np.linspace(-L,L,int((2*L)/h+1)*1000) #x continu
print(y)
print(x)



def u1(X,L):
    res=[]
    for x in X:
        if (x==0):
            res.append(1)
            
        else:
            res.append(0)
    return res

def u2(X,L):
    res=[]
    for x in X:
        if (abs(x)<=(L/4)):
            res.append(1)
        else:
            res.append(0)
    return res

def u3(X,L):
    res=[]
    for x in X:
        if (x <=0. ) and (x> (-L/3)):
            res.append(3./(L*(x+L/3)))
        elif (x>0.) and (x <=(L/3)):
            res.append(1-3./(L*x))
        else:
            res.append(0)
    return res

#def Sh(X,h):
#    res=[]
#    for x in X:
#        res.append(np.sin(np.pi*x/h)/(np.pi*x/h))
#    return res


def p1(X,L,h):
    x=u1(X,L)
   # S=Sh(X,h)
    p=np.zeros(len(X))
    for i in range (0,len(X)-1):
        for m in range (0,int(len(X)-1)):
            xm=h*(m-L)
            if x[i]==xm:
                p[i]=p[i]+x[i]
            else:
                p[i]=p[i]+x[i]*(np.sin(np.pi*(x[i]-xm/h)/(np.pi*(x[i]-xm)/h)))
    return p

def p2(X,xm,L,h):
    x=u2(X,L)
    p=[]
#    S=Sh(X,h)
#    cpt=0.0
    for i in range (0,len(x)-1):
        s=0
        for m in range (0,len(xm)-1):
            if x[i]==xm[m]:
                p[i]=p[i]+x[i]
            else:
                s=s+x[i]*(np.sin(np.pi*(x[i]-xm[m]/h)/(np.pi*(x[i]-xm[m])/h)))
                p.append(s)
    return p

def p3(X,L,h):
    x=u3(X,L)
    p=np.zeros(len(X))
#    S=Sh(X,h)
#    cpt=0.0
    for i in range (0,len(X)-1):
        for m in range (0,int(len(X)-1)):
            xm=h*(m-L)
            if x[i]==xm:
                p[i]=p[i]+x[i]
            else:
                p[i]=p[i]+x[i]*(np.sin(np.pi*(x[i]-xm/h)/(np.pi*(x[i]-xm)/h)))
    return p



x_grille=[]
y_grille=[]
for i in range (0,len(x)):
    for j in range (0,len(y)):
        x_grille.append(x[i])
        y_grille.append(y[j])
    
plt.scatter(x_grille,y_grille,color="red")
plt.show()

plt.plot(x,u1(x,L), color="blue")
plt.grid(color="red")
plt.show()

plt.plot(x,p1(x,L,h))
plt.show()


plt.plot(x,u2(x,L), color="blue")
plt.grid(color="red")
plt.show()

plt.plot(x,p2(x,x_m,L,h))
plt.show()


plt.plot(x,u3(x,L), color="blue")
plt.grid(color="red")
plt.show()

plt.plot(x,p3(x,L,h))
plt.show()



#%%
def DerSpecPer(x):
    return 0








