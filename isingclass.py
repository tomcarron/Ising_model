# -*- coding: utf-8 -*-
"""
Ising script for import
object oriented approach
"""
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Ising:
   
    def __init__(self,x,y,T,H,spins=[-1,1],J=-1,kb=1.0,mu=1):
        ''' Initialise a random lattice '''
        self.x = x
        self.y = y
        self.T = T
        self.H = H
        self.spins = spins
        self.J = J
        self.kb = kb
        self.mu = mu
        
        self.initial_lattice = np.random.choice(spins,(x, y))
        self.lattice = np.copy(self.initial_lattice)
        

    #Nearest neighbours with periodic boundary conditions
    def nearest_neighbours(self,candidate,i2,j2):
        N,M=candidate.shape
        #top
        if j2-1 < 1.0:
            qt=M-1
            pt=i2
        else:
            qt=j2-1
            pt=i2
        top=int(candidate[pt,qt])
        #bottom
        if j2+1 >= M:
            qb=1
            pb=i2
        else:
            qb=j2+1
            pb=i2
        bottom=int(candidate[pb,qb])
        #left
        if i2-1 < 1:
            pl=N-1
            ql=j2
        else:
            pl=i2-1
            ql=j2
        left=int(candidate[pl,ql])
        #right
        if i2+1 >= N:
            pr=1
            qr=j2
        else:
            pr=i2+1
            qr=j2
        right=int(candidate[pr,qr])
        
        neighbours=[top,bottom,left,right]
        return neighbours
    #determines wether or not the spin flips, according to dE and boltzmann factor.
    def spin_flip(self,candidate,i2,j2):
        neighbours = self.nearest_neighbours(candidate,i2,j2)
        deltaE=-2*(candidate[i2,j2])*((self.J)*(np.sum(neighbours))+(self.H))
        if deltaE<=0:
            candidate[i2,j2]*=-1
        elif np.e**((-1.0*deltaE)/(self.kb*self.T))>np.random.rand():
            candidate[i2,j2]*=-1
    
    
    def sweep(self):
        '''one updating sweep of system'''
        candidate=np.copy(self.lattice)
        x,y=candidate.shape
        for i_offset in range(2):
            for j_offset in range(2):
                for i2 in range(i_offset, x, 2):
                    for j2 in range(j_offset, y, 2):
                        self.spin_flip(candidate,i2,j2)
        self.lattice=np.copy(candidate)
        
    def update(self,num):
        '''run a number of sweeps of system'''
        for i in range(num):
            self.sweep()
    
    def energy(self):
        '''energy is the sum over all pairs of nearest neighbours
            of the product of J,spin i and spin j '''
        energy=0.0
        x,y=self.lattice.shape
        for i in range(x):
            for j in range(y):
                energy+= (-self.J)*((self.lattice[i,j])*(np.sum(self.nearest_neighbours(self.lattice,i,j)))) - (self.mu)*(self.H)*(self.lattice[i,j])
        energy=energy/(x*y)
        return energy
    
    def magnetization(self):
        '''closely related to the average spin alignment,
            for an infinitely large, spins will all have
            same average alignment, for our system, sum of all states'''  
            
        magnet=0.0
        x,y=self.lattice.shape
        for i in range(x):
            for j in range(y):
                magnet+=self.lattice[i,j]
        magnet=magnet/(x*y)
        return magnet
  
    def cv(self):
        '''Specific heat capacity'''
        sqavg=0.0
        avgsq=0.0
        x,y=self.lattice.shape
        for i in range(x):
            for j in range(y):
                sqavg+= ((-self.J)*((self.lattice[i,j])*(np.sum(self.nearest_neighbours(self.lattice,i,j)))) - (self.mu)*(self.H)*(self.lattice[i,j]))**2
                avgsq+= (-self.J)*((self.lattice[i,j])*(np.sum(self.nearest_neighbours(self.lattice,i,j)))) - (self.mu)*(self.H)*(self.lattice[i,j])
        sq_avg=sqavg/(x*y)
        avg_sq=((avgsq)/(x*y))**2
        cv=((sq_avg-avg_sq)/((self.kb)*(self.T)*(self.T))) / (x*y)
        return cv
    
    def mag_sus(self):
        '''Magnetic susceptibility''' 
        m_sqavg=0.0
        m_avgsq=0.0
        x,y=self.lattice.shape
        for i in range(x):
            for j in range(y):
                m_sqavg+=(self.lattice[i,j])**2
                m_avgsq+=self.lattice[i,j]
        m_sq_avg=m_sqavg/(x*y)
        m_avg_sq=((m_avgsq)/(x*y))**2
        mag_sus=((m_sq_avg-m_avg_sq)/((self.kb)*(self.T)))
        return mag_sus
        
    def plot_temp(self,num,sweeps):
        '''performs simulation over varying 
           temperatures and plots results'''
        temp=np.arange(0.1,5.1,0.1)
        bigenergylist=np.zeros_like(temp)
        bigmagnetlist=np.zeros_like(temp)
        bigtemplist=np.zeros_like(temp)
        temp2=np.arange(1.0,5.1,0.1)
        bigcvlist=np.zeros_like(temp2)
        bigmagsuslist=np.zeros_like(temp2)
        bigtemp2list=np.zeros_like(temp2)
        x,y=self.lattice.shape
        H=self.H
        for i in range(num):
            T=0.1
            model=Ising(x,y,T,H)
            i=0
            while T < 5.0:
                i2=i-9
                model.update(sweeps)
                bigenergylist[i]+=(model.energy())
                bigmagnetlist[i]+=abs(model.magnetization())
                bigtemplist[i]+=(T)
                if T > 0.99:
                    bigcvlist[i2]+=model.cv()
                    bigmagsuslist[i2]+=(model.mag_sus())
                    bigtemp2list[i2]+=(T)
                T+=0.1
                i+=1
                model=Ising(x,y,T,H)
        bigenergy=np.array(bigenergylist)/num
        bigmagnet=np.array(bigmagnetlist)/num
        bigmagsus=np.array(bigmagsuslist)/num
        bigtemp=np.array(bigtemplist)/num
        bigcv=np.array(bigcvlist)/num
        bigtemp2=np.array(bigtemp2list)/num
        fig1=plt.figure(1)
        ax1=fig1.add_subplot(1,1,1)
        ax1.set_aspect('auto')
        ax1.set_title('Magnetization per spin vs Temperature')
        ax1.grid()
        ax1.set_xlabel('Temperature (J/Kb)')
        ax1.set_ylabel('Average Magnetization per spin')
        ax1.plot(bigtemp,bigmagnet,'bo',markersize=2)
        fig2=plt.figure(2)
        ax2=fig2.add_subplot(1,1,1)
        ax2.set_aspect('auto')
        ax2.set_title('Average energy per spin vs Temperature')
        ax2.grid()
        ax2.set_xlabel('Temperature (J/Kb)')
        ax2.set_ylabel('Average energy per spin')
        ax2.plot(bigtemp,bigenergy,'bo',markersize=2)
        fig3=plt.figure(3)
        ax3=fig3.add_subplot(1,1,1)
        ax3.set_aspect('auto')
        ax3.set_title('Specific heat vs Temperature')
        ax3.grid()
        ax3.set_xlabel('Temperature (J/Kb)')
        ax3.set_ylabel('Specific heat per spin')
        ax3.plot(bigtemp2,bigcv,'bo',markersize=2)
        fig4=plt.figure(4)
        ax4=fig4.add_subplot(1,1,1)
        ax4.set_aspect('auto')
        ax4.set_title('Magnetic susceptibility vs Temperature')
        ax4.grid()
        ax4.set_xlabel('Temperature (J/Kb)')
        ax4.set_ylabel('Magnetic susceptibility per spin')
        ax4.plot(bigtemp2,bigmagsus,'bo',markersize=2)

    def plot_magfield(self,num,sweeps,T):
        '''performs varying H simulation and plots results for a given temperature'''
        x,y=self.lattice.shape
        model=Ising(x,y,T,-5)
        hrange=np.arange(-5,5.5,0.5)
        magnetlist=np.zeros_like(hrange)
        for i in range(num):
            H=-5
            model=Ising(x,y,T,H)
            i=0
            while H < 5.1:
                model.update(sweeps)
                magnetlist[i]+=model.magnetization()
                H+=0.5
                i+=1
                model=Ising(x,y,T,H)
        magnet=np.array(magnetlist)/num
        fig5=plt.figure(5)
        ax5=fig5.add_subplot(1,1,1)
        ax5.set_aspect('auto')
        ax5.set_title('T='+str(T))
        ax5.grid()
        ax5.set_xlabel('H')
        ax5.set_ylabel('M')
        ax5.plot(hrange,magnet,'bo',markersize=2)

    def animate(self,frames,save_name):
        '''performs animation of model evolution'''
        fig6 = plt.figure(6)
        ax6=fig6.add_subplot(1,1,1)
        ax6.set_aspect('equal')
        ax6.set_title('Ising Model')
        ax6.set_xticks([])                                                     
        ax6.set_yticks([])
        ax6.imshow(self.initial_lattice, interpolation='nearest', cmap=cm.summer_r)
        def animate(frames,model):
            if frames != 0:
                model.sweep()
            return ax6.imshow(model.lattice, interpolation='nearest', cmap=cm.summer_r)
        
        ani=FuncAnimation(fig6 , animate, frames=frames, interval=200, repeat=True, blit=False, fargs=[self]) 
        ani.save(str(save_name)+'.gif', writer='imagemagick', fps=5)
        
    def print_still(self):
        '''prints a still image of lattice macrostate'''
        fig7=plt.figure(7)
        ax7=fig7.add_subplot(1,1,1)
        ax7.set_aspect('equal')
        ax7.set_xticks([])                                                     
        ax7.set_yticks([])
        ax7.imshow(self.lattice, interpolation='nearest', cmap=cm.summer_r)

plt.show()