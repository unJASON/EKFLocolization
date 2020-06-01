import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from light_house_create import dataCreate

from SwarmEKF2 import swarmEKF

show_animation = True
np.random.seed(16) # seed the random number generator for reproducibility
border = {"xmin":-8, "xmax":8, "ymin":-8, "ymax":8,"zmin":0,"zmax":5}
numRob = 10 # number of robots
lighthouse_Idx=[0,1]
dimension = 2
dt = 0.01 # time interval [s]
simTime = 70.0 # simulation time [s]
maxVel = 2 # maximum velocity [m/s]
devInput = np.array([[0.25, 0.25, 0.01]]).T # input deviation in simulation, Vx[m/s], Vy[m/s], yawRate[rad/s]
devObser = 0.1 # observation deviation of distance[m]
ekfStride = 1 # update interval of EKF is simStride*0.01[s]

#初始化
xTrue = np.random.uniform(-5, 5, (dimension, numRob)) # random initial groundTruth of state [x, y, yaw]' of numRob robots
relativeState = np.zeros((3, numRob, numRob)) # [x_ij, y_ij, yaw_ij]' of the second robot in the first robot's view
data = dataCreate(numRob, border, maxVel, dt, devInput, devObser,dimension=dimension) #Create input data such as velocities, yaw rates, distances...
estiEKF = swarmEKF(10, 0.1, 0.25, 0.4, 0.1, numRob,dt=dt,dimension=dimension,lighthouse_Idx=lighthouse_Idx)

xTrue[:,0] = [0,0]
xTrue[:,1] = [1,1]
xEsti = np.random.uniform(-5, 5,(dimension*2, numRob))
# xEsti[dimension:,:] = 0
# xEsti[:dimension,:] = xTrue
def goRec(u,step):
    cycle = 1500
    vel = -1
    if step % cycle < cycle * 0.25:
        u[:, 1] = [vel, 0]
    elif step % cycle < cycle * 0.5:
        u[:, 1] = [0, vel]
    elif step % cycle < cycle * 0.75:
        u[:, 1] = [-vel, 0]
    else:
        u[:, 1] = [0, -vel]
    u[:, 0] = -u[:, 1]
    # u[:, 0] = 0
    # u[:, 1] = 0
    # u[:, 2] = 0
    u[:,2:] = 0
    return u
def animate(step):
    global xTrue, relativeState, xEsti,estiEKF,Pmatrix
    u = data.calcInput_FlyIn1m(step)
    # u = goRec(u,step)

    xTrue, zNois, uNois = data.update(xTrue, u)

    if step % ekfStride == 0:
        #生成一个随机排列 然后作为后者的某个列表加入
        # permu = np.random.permutation(numRob).tolist()
        # for ele in lighthouse_Idx:
        #     permu.remove(ele)
        # permu = lighthouse_Idx + permu

        reference_list = []
        for i in range(numRob):
            xEsti = estiEKF.CovEKF(uNois, zNois, xEsti, xTrue, ekfStride, i)
            # xEsti = estiEKF.CovEKF2(uNois, zNois, xEsti, xTrue, ekfStride, i, reference_list)
            # reference_list.append(i)
            # print(i, estiEKF.Pmatrix[i, 0, 0],estiEKF.Pmatrix[i, 1, 1],estiEKF.Pmatrix[i, 2, 2],estiEKF.Pmatrix[i, 3, 3])
            # print(i,"true:", xTrue[:,i], "est:",xEsti[0:2, i]," ",xEsti[0+dimension:2+dimension,i])
    pointsTrue.set_data(xTrue[0, :], xTrue[1, :])  # plot groundTruth points
    pointsEsti.set_data(xEsti[0, :]+xEsti[0+dimension,:], xEsti[1, :]+xEsti[1+dimension,:])  # plot estimated points


    circle.center = (xTrue[0, 0], xTrue[1, 0])
    circle.radius = zNois[0, 1]  # plot a circle to show the distance between robot 0 and robot 1

    time_text.set_text("t={:.2f}s".format(step * dt))
    return pointsTrue, pointsEsti, circle, pointsTrueHead, pointsEstiHead, time_text


def animate3D(step):
    global xTrue, relativeState, xEsti
    u = data.calcInput_FlyIn1m(step)
    xTrue, zNois, uNois = data.update(xTrue, u)

    if step % ekfStride == 0:
        for i in range(numRob):
            # xEsti = estiEKF.EKF(uNois, zNois, xEsti,xTrue, ekfStride, i)
            xEsti = estiEKF.CovEKF(uNois, zNois, xEsti, xTrue, ekfStride, i)
            # xEsti = estiEKF.anchorEKF(uNois, zNois, xEsti, xTrue, ekfStride, i)
    pointsTrue = ax.scatter(xTrue[0, :], xTrue[1, :],xTrue[2,:], c="b")  # plot groundTruth points
    pointsEsti = ax.scatter(xEsti[0, :], xEsti[1, :],xEsti[2,:], c="r")  # plot estimated points
    return pointsTrue, pointsEsti

if show_animation:
    if dimension == 2:
        # Set up an animation
        fig = plt.figure()
        ax  = fig.add_subplot(111, aspect='equal')
        ax.set(xlim=(border["xmin"], border["xmax"]), ylim=(border["ymin"], border["ymax"]))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        title = ax.set_title('Simulated swarm')
        pointsTrue,  = ax.plot([], [], linestyle="", marker="o", color="b", label="GroundTruth")
        pointsEsti,  = ax.plot([], [], linestyle="", marker="o", color="r", label="Relative EKF")
        pointsTrueHead,  = ax.plot([], [], linestyle="", marker=".", color="g")
        pointsEstiHead,  = ax.plot([], [], linestyle="", marker=".", color="g")
        ax.legend()
        circle = plt.Circle((0, 0), 0.2, color='black', fill=False)
        ax.add_patch(circle)

        rectangle = plt.Rectangle((-5,-5),width=10,color='pink',height=8,fill=False)
        ax.add_patch(rectangle)

        time_text = ax.text(0.01, 0.97, '', transform=ax.transAxes)
        time_text.set_text('')
        #ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
        ani = animation.FuncAnimation(fig, animate,init_func=None, frames=None, interval=1, blit=True)
        plt.show()
    elif dimension == 3:
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.set_xlim3d([border["xmin"], border["xmax"]])
        ax.set_xlabel('X')
        ax.set_ylim3d([border["ymin"], border["ymax"]])
        ax.set_ylabel('Y')
        ax.set_zlim3d([border["zmin"], border["zmax"]])
        ax.set_zlabel('Z')
        ax.set_title('Simulated swarm')

        # pointsTrueHead, = ax.plot([], [], linestyle="", marker=".", color="g")
        # pointsEstiHead, = ax.plot([], [], linestyle="", marker=".", color="g")
        ax.legend()

        # time_text = ax.text(0.01, 0.97, '', transform=ax.transAxes)
        # time_text.set_text('')
        ani = animation.FuncAnimation(fig, animate3D, frames=None, interval=100, blit=True)
        plt.show()
    else:
        pass
else:
   pass