import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from light_house_create import dataCreate
import SwarmEKF
import SwarmEKF2

show_animation = True
np.random.seed(1556446) # seed the random number generator for reproducibility
border = {"xmin":-8, "xmax":8, "ymin":-8, "ymax":8,"zmin":0,"zmax":5}
numRob = 10 # number of robots
lighthouse_Idx=[0,1,2]
dimension = 2
dt = 0.01 # time interval [s]
simTime = 10.0 # simulation time [s]
maxVel = 4 # maximum velocity [m/s]
devInput = np.array([[0.25, 0.25, 0.01]]).T # input deviation in simulation, Vx[m/s], Vy[m/s], yawRate[rad/s]
devObser = 0.1 # observation deviation of distance[m]
ekfStride = 1 # update interval of EKF is simStride*0.01[s]

#初始化
xTrue = np.random.uniform(-5, 5, (dimension, numRob)) # random initial groundTruth of state [x, y, yaw]' of numRob robots
relativeState = np.zeros((3, numRob, numRob)) # [x_ij, y_ij, yaw_ij]' of the second robot in the first robot's view
data = dataCreate(numRob, border, maxVel, dt, devInput, devObser,dimension=dimension) #Create input data such as velocities, yaw rates, distances...

estiEKF1 = SwarmEKF.swarmEKF(10, 0.1, 0.25, 0.4, 0.1, numRob,dt=dt,dimension=dimension,lighthouse_Idx=lighthouse_Idx)
estiEKF2 = SwarmEKF2.swarmEKF(10, 0.1, 0.25, 0.4, 0.1, numRob,dt=dt,dimension=dimension,lighthouse_Idx=lighthouse_Idx)
estiEKF3 = SwarmEKF2.swarmEKF(10, 0.1, 0.25, 0.4, 0.1, numRob,dt=dt,dimension=dimension,lighthouse_Idx=lighthouse_Idx)

# xTrue[:,0] = [0,0]
# xTrue[:,1] = [1,1]


xEsti1 = np.random.uniform(-5, 5,(dimension, numRob))
xEsti2 = np.random.uniform(-5, 5,(dimension*2, numRob))
xEsti3 = np.random.uniform(-5, 5,(dimension*2, numRob))
xEsti1[:,:] = 0
xEsti2[:,:] = 0
xEsti3[:,:] = 0
# xEsti3 = xEsti2[:,:]
# xEsti1 = xEsti2[:2,:]

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
    global xTrue, relativeState, xEsti1,xEsti2,xEsti3,estiEKF,Pmatrix
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
            xEsti1 = estiEKF1.CovEKF(uNois, zNois, xEsti1, xTrue, ekfStride, i)
            xEsti2 = estiEKF2.CovEKF(uNois, zNois, xEsti2, xTrue, ekfStride, i)
            xEsti3 = estiEKF3.CovEKF2(uNois, zNois, xEsti3, xTrue, ekfStride, i,u)
            # xEsti = estiEKF.CovEKF2(uNois, zNois, xEsti, xTrue, ekfStride, i, reference_list)
            # reference_list.append(i)
            # print(i, estiEKF.Pmatrix[i, 0, 0],estiEKF.Pmatrix[i, 1, 1],estiEKF.Pmatrix[i, 2, 2],estiEKF.Pmatrix[i, 3, 3])
            # print(i,"true:", xTrue[:,i], "est:",xEsti[0:2, i]," ",xEsti[0+dimension:2+dimension,i])


    pointsTrue.set_data(xTrue[0, :], xTrue[1, :])  # plot groundTruth points
    pointsTrue_ax1.set_data(xTrue[0, :], xTrue[1, :])  # plot groundTruth points
    pointsTrue_ax2.set_data(xTrue[0, :], xTrue[1, :])  # plot groundTruth points
    pointsTrue_ax3.set_data(xTrue[0, :], xTrue[1, :])  # plot groundTruth points

    pointsEsti1.set_data(xEsti1[0, :],xEsti1[1, :])  # plot estimated points
    pointsEsti2.set_data(xEsti2[0, :]+xEsti2[0+dimension,:], xEsti2[1, :]+xEsti2[1+dimension,:])
    pointsEsti3.set_data(xEsti3[0, :] + xEsti3[0 + dimension, :],xEsti3[1, :] + xEsti3[1 + dimension, :])


    pointsEsti_ax1.set_data(xEsti1[0, :], xEsti1[1, :])  # plot estimated points
    pointsEsti_ax2.set_data(xEsti2[0, :] + xEsti2[0 + dimension, :], xEsti2[1, :] + xEsti2[1 + dimension, :])
    pointsEsti_ax3.set_data(xEsti3[0, :] + xEsti3[0 + dimension, :], xEsti3[1, :] + xEsti3[1 + dimension, :])


    circle.center = (xTrue[0, 0], xTrue[1, 0])
    circle.radius = zNois[0, 1]  # plot a circle to show the distance between robot 0 and robot 1

    time_text.set_text("t={:.2f}s".format(step * dt))
    return pointsTrue, pointsEsti1,pointsEsti2,pointsEsti3, circle, time_text,pointsTrue_ax1,pointsTrue_ax2,pointsTrue_ax3,pointsEsti_ax1,pointsEsti_ax2,pointsEsti_ax3

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
        fig.set_size_inches(6, 6)
        ax = fig.add_subplot(221, aspect='equal')
        ax.set(xlim=(border["xmin"], border["xmax"]), ylim=(border["ymin"], border["ymax"]))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        title = ax.set_title('Simulated swarm')
        pointsTrue,  = ax.plot([], [], linestyle="", marker="o", color="b", label="GroundTruth")
        pointsEsti1,  = ax.plot([], [], linestyle="", marker="o", color="r", label="EKF1")
        pointsEsti2, = ax.plot([], [], linestyle="", marker="o", color="g", label="EKF2")
        pointsEsti3, = ax.plot([], [], linestyle="", marker="o", color="y", label="EKF3")


        circle = plt.Circle((0, 0), 0.2, color='black', fill=False)
        ax.add_patch(circle)
        rectangle = plt.Rectangle((-5,-5),width=10,color='pink',height=8,fill=False)
        ax.add_patch(rectangle)
        time_text = ax.text(0.01, 0.01, '', transform=ax.transAxes)
        time_text.set_text('')
        #ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
        ani = animation.FuncAnimation(fig, animate,init_func=None, frames=None, interval=1, blit=True)

        ax1 = fig.add_subplot(222, aspect='equal')
        ax1.set(xlim=(border["xmin"], border["xmax"]), ylim=(border["ymin"], border["ymax"]))
        pointsTrue_ax1, = ax1.plot([], [], linestyle="", marker="o", color="b")
        pointsEsti_ax1, = ax1.plot([], [], linestyle="", marker="o", color="r")

        ax2 = fig.add_subplot(223, aspect='equal')
        ax2.set(xlim=(border["xmin"], border["xmax"]), ylim=(border["ymin"], border["ymax"]))
        pointsTrue_ax2, = ax2.plot([], [], linestyle="", marker="o", color="b")
        pointsEsti_ax2, = ax2.plot([], [], linestyle="", marker="o", color="g")

        ax3 = fig.add_subplot(224, aspect='equal')
        ax3.set(xlim=(border["xmin"], border["xmax"]), ylim=(border["ymin"], border["ymax"]))
        pointsTrue_ax3, = ax3.plot([], [], linestyle="", marker="o", color="b")
        pointsEsti_ax3, = ax3.plot([], [], linestyle="", marker="o", color="y")
        fig.legend(loc='upper right')

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
    xEsti1[:, :] = 0
    xEsti2[:, :] = 0
    xEsti1 = xTrue
    xEsti2[:2, :] = xTrue
    droneID = 2
    dataForPlot1 = np.array([xEsti1[0, droneID], xTrue[0, droneID],
                            xEsti1[1, droneID], xTrue[1, droneID], ])  #
    dataForPlot2 = np.array([xEsti2[0, droneID], xTrue[0, droneID],
                             xEsti2[1, droneID], xTrue[1, droneID], ])  #

    step = 0
    while simTime >= dt * step:
        step += 1
        u = data.calcInput_FlyIn1m(step)
        # u = data.calcInput_PotentialField(step, xTrue)
        # u = data.calcInput_Formation01(step, relativeState)
        # u = data.calcInput_FlyIn1mRob1NoVel(step)
        xTrue, zNois, uNois = data.update(xTrue, u)
        if step % ekfStride == 0:
            for i in range(numRob):
                xEsti1 = estiEKF1.CovEKF(uNois, zNois, xEsti1, xTrue, ekfStride, i)
                xEsti2 = estiEKF2.CovEKF(uNois, zNois, xEsti2, xTrue, ekfStride, i)
                # xEsti2 = estiEKF3.CovEKF2(uNois, zNois, xEsti2, xTrue, ekfStride, i, u)
        dataForPlot1 = np.vstack([dataForPlot1, np.array(
            [xEsti1[0, droneID], xTrue[0, droneID],
             xEsti1[1, droneID], xTrue[1, droneID],
             ])])
        dataForPlot2 = np.vstack([dataForPlot2, np.array(
        [xEsti2[0, droneID]+xEsti2[0+dimension,droneID], xTrue[0, droneID],
         xEsti2[1, droneID]+xEsti2[1+dimension,droneID], xTrue[1, droneID],
         ])])
    error1_x = np.fabs( dataForPlot1[:, 0] - dataForPlot1[:, 1] ).sum()/dataForPlot1.shape[0]
    error1_y = np.fabs( dataForPlot1[:, 2] - dataForPlot1[:, 3]).sum()/dataForPlot1.shape[0]
    error2_x = np.fabs( dataForPlot2[:, 0] - dataForPlot2[:, 1]).sum()/dataForPlot1.shape[0]
    error2_y = np.fabs( dataForPlot2[:, 2] - dataForPlot2[:, 3]).sum()/dataForPlot1.shape[0]
    print(error1_x,error1_y)
    print(error2_x, error2_y)
    dataForPlotArray = dataForPlot1.T
    timePlot = np.arange(0, len(dataForPlotArray[0])) / 100
    f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, sharex=True)
    plt.margins(x=0)
    ax1.plot(timePlot, dataForPlotArray[0, :])
    ax1.plot(timePlot, dataForPlotArray[1, :])
    ax1.set_ylabel(r"$x_{ij}$ (m)", fontsize=12)
    ax1.grid(True)
    ax2.plot(timePlot, dataForPlotArray[2, :])
    ax2.plot(timePlot, dataForPlotArray[3, :])
    ax2.set_ylabel(r"$y_{ij}$ (m)", fontsize=12)
    ax2.grid(True)

    dataForPlotArray = dataForPlot2.T
    ax3.plot(timePlot, dataForPlotArray[0, :])
    ax3.plot(timePlot, dataForPlotArray[1, :])
    ax3.set_ylabel(r"$x_{ij}$ (m)", fontsize=12)
    ax3.grid(True)
    ax4.plot(timePlot, dataForPlotArray[2, :])
    ax4.plot(timePlot, dataForPlotArray[3, :])
    ax4.set_ylabel(r"$y_{ij}$ (m)", fontsize=12)
    ax4.grid(True)


    # Fine-tune figure; make subplots close to each other and hide x ticks for all but bottom plot.
    f.subplots_adjust(wspace=0.5,hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()