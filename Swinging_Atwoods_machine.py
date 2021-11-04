import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as smp
from matplotlib import animation
from matplotlib.animation import PillowWriter
import math

t, g, M, m = smp.symbols('t, g, M, m')
r, theta = smp.symbols(r'r \theta', cls=smp.Function)

theta = theta(t)
r = r(t)
dtheta = smp.diff(theta, t)
dr = smp.diff(r, t)
ddtheta = smp.diff(dtheta, t)
ddr = smp.diff(dr, t)

T = (1/2)*M*dr**2 +  (1/2)*m*(dr**2 + r**2*dtheta**2)
U = M*g*r - m*g*r*smp.cos(theta)
L = T - U

LE1 = smp.diff(L, theta) - smp.diff(smp.diff(L, dtheta), t).simplify()
LE2 = smp.diff(L, r)     - smp.diff(smp.diff(L, dr), t).simplify()

sols = smp.solve([LE1, LE2], (ddtheta, ddr), simplify=False, rational=False)

dz1dt_f = smp.lambdify((t, g, M, m,theta,r,dtheta, dr), sols[ddtheta])
dz2dt_f = smp.lambdify((t, g, M, m,theta,r,dtheta, dr), sols[ddr])
dthetadt_f = smp.lambdify(dtheta, dtheta)
drdt_f = smp.lambdify(dr, dr)

def dSdt(S, t, g, M, m):
    theta, z1, r, z2 = S
    return [
        dthetadt_f(z1),
        dz1dt_f(t, g, M, m,theta,r,z1,z2),
        drdt_f(z2),
        dz2dt_f(t, g, M, m,theta,r,z1,z2),
    ]

#Generalize for differents mu

def simulate_orbits(mu, t_end, steps):
    t = np.linspace(0, t_end, steps) # s

    # mu = M/m
    g = 9.8
    m = 1
    M = m*mu 

    r_0 = 1.0
    dr_0 = 0.0
    theta_0 = math.pi/2
    dtheta_0 = 0.0

    return odeint(dSdt, y0=[theta_0,dtheta_0, r_0, dr_0 ], t=t, args=(g, M, m))



def get_x1y1_x2y2(data):
    # x1y1 m position
    # x2y2 M position
    
    L = 1
    theta = data.T[0]
    r = data.T[2]

    return (L + r*np.sin(theta), -r*np.cos(theta), 0*r, -2+r)


def create_video(mu):
    #mu = 4
    t_end = 100
    steps = 10000

    data = simulate_orbits(mu,t_end,steps)
    x1, y1, x2, y2 = get_x1y1_x2y2(data)


    def animate(i):
        ln1.set_data([x1[i], y1[i]])
        ln2.set_data([x2[i], y2[i]])
        ln3.set_data([0,x2[i]], [0,y2[i]])
        ln4.set_data([1,x1[i]], [0,y1[i]])
        ln5.set_data([x1[0:i], y1[0:i]])

    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.set_facecolor('k')
    ax.get_xaxis().set_ticks([])    # enable this to hide x axis ticks
    ax.get_yaxis().set_ticks([])    # enable this to hide y axis ticks

    mass_m_size = 8
    mass_M_size =  mass_m_size*mu
    plt.plot([0,1], [0,0], 'r-', lw=3)
    ln1, = plt.plot([], [], 'ro--', lw=3, markersize=mass_m_size)
    ln2, = plt.plot([], [], 'ro--', lw=3, markersize=mass_M_size)
    ln3, = plt.plot([], [], 'r-', lw=3)
    ln4, = plt.plot([], [], 'r-', lw=3)
    ln5, = plt.plot([], [], 'w-', lw=1)

    ax.set_ylim(-2,1)
    ax.set_xlim(-0.5,2.5)
    ani = animation.FuncAnimation(fig, animate, frames=2000, interval=25)
    #ani.save('Swinging_Atwoods_Machine.gif',writer='pillow',fps=25)


    name_file = 'Swinging_Atwoods_Machine_' + str(mu) +'.mp4'
    FFwriter = animation.FFMpegWriter(fps=25, extra_args=['-vcodec', 'libx264'])
    ani.save(name_file, writer=FFwriter)
    
# Create de video    
create_video(mu=4)
