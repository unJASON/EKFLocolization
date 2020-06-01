"""
Create input data such as velocities, yaw rates, distances...
"""
import numpy as np

class dataCreate:

    def __init__(self, numRob, border, maxVel, dt, devInput, devObser,dimension = 2):
        self.velocity = np.zeros((dimension, numRob))
        self.numRob = numRob
        self.border = border
        self.maxVel = maxVel
        self.dt = dt
        self.dimension = dimension
        self.devInput = devInput[:dimension]
        self.devObser = devObser
        # variables to be used in PID formation control
        self.intErrX = 0
        self.oldErrX = 0
        self.intErrY = 0
        self.oldErrY = 0

    def calcInput_PotentialField(self, step, xTrue):
    # Calculate control inputs [vx, vy, yaw_rate]' of all robots using Potential Field method
    # such that all robots fly randomly within the border and avoid each other
        if (step % 500 == 0):
            # Create random [vx, vy, yaw_rate]' that sustain for 5 seconds
            # The range is [-1, 1]m and [-1, 1]rad/s
            self.velocity[0:2, :] = np.random.uniform(0, self.maxVel*2, (2, self.numRob)) - self.maxVel
            self.velocity[2, :] = np.random.uniform(-1, 1, (1, self.numRob))
        # Reverse the velocity when reaching the border
        for i in range(self.numRob):
            if self.border["xmax"]-xTrue[0,i]<1.0:
                self.velocity[0, i] = -abs(self.velocity[0, i])
            if xTrue[0,i]-self.border["xmin"]<1.0:
                self.velocity[0, i] = abs(self.velocity[0, i])
            if self.border["ymax"]-xTrue[1,i]<1.0:
                self.velocity[1, i] = -abs(self.velocity[1, i])
            if xTrue[1,i]-self.border["ymin"]<1.0:
                self.velocity[1, i] = abs(self.velocity[1, i])
        # Collision avoidance
        velocity_avoidance = np.zeros((3, self.numRob))
        for i in range(self.numRob):
            [vx, vy] = [0, 0]
            posI = [xTrue[0,i], xTrue[1,i]]
            for j in range(self.numRob):
                if j!= i:
                    posJ = [xTrue[0,j], xTrue[1,j]]
                    distIJ = np.sqrt((posI[0]-posJ[0])**2+(posI[1]-posJ[1])**2)
                    if distIJ < 1:
                        vx = vx + 1/(posI[0]-posJ[0])
                        vy = vy + 1/(posI[1]-posJ[1])
            velocity_avoidance[0:2,i] = 0.15*np.clip(np.array([vx, vy]).T, -10, 10)
        velocity_temp = self.velocity+velocity_avoidance
        velocity_output = self.velocity+velocity_avoidance
        # Rotate the velocity_temp from earth-frame to body frame
        velocity_output[0,:] =  velocity_temp[0,:] * np.cos(xTrue[2,:]) + velocity_temp[1,:] * np.sin(xTrue[2,:])
        velocity_output[1,:] = -velocity_temp[0,:] * np.sin(xTrue[2,:]) + velocity_temp[1,:] * np.cos(xTrue[2,:])
        return velocity_output

    def calcInput_FlyIn1m(self, step):
    # Calculate control inputs [vx, vy, yaw_rate]' of all robots such that
    # all robots fly randomly within 1m range
        if (step % 100)== 0:
            if (step % 200) == 0:
                self.velocity = -self.velocity
            else:
                if self.dimension == 3:
                    self.velocity[0:2,:] = np.random.uniform(0, self.maxVel*2, (2, self.numRob)) - self.maxVel
                    self.velocity[2,:] = np.random.uniform(0, 1, (1, self.numRob)) - 0.5
                else:
                    self.velocity[0:2, :] = np.random.uniform(0, self.maxVel * 2, (2, self.numRob)) - self.maxVel
        return self.velocity

    def motion_model(self, x, u):
        # Robot model for state prediction
        xPred = np.zeros((self.dimension, self.numRob))
        # F = np.array([[1.0, 0, 0],
        #               [0, 1.0, 0],
        #               [0, 0, 1.0]])
        # B = np.array([[1.0, 0, 0],
        #               [0, 1.0, 0],
        #               [0, 0, 1.0]]) * self.dt
        F = np.diag([1.0]*self.dimension)
        B = np.diag([1.0*self.dt]*self.dimension)
        for i in range(self.numRob):
            # X_{k+1} = X_k + Ve * dt; e means earth, b means body
            # Ve = [[c(psi), -s(psi)],[s(psi),c(psi)]] * Vb
            xPred[:,i] = F@x[:,i] + B@u[:,i]
        return xPred
    def update(self, xTrue, u):  #u=xEsti
        # Calculate the updated groundTruth(xTrue), noised observation(zNoise), and noised input(uNoise)
        # xTrue = self.motion_model(xTrue, u)
        xTrue = self.motion_model(xTrue, u)

        zTrue = np.zeros((self.numRob, self.numRob)) # distances
        for i in range(self.numRob):
            for j in range(self.numRob):
                for k in range(self.dimension):
                    dk = xTrue[k,i] -xTrue[k,j]
                    zTrue[i, j] = zTrue[i, j] + dk**2
                zTrue[i,j] = np.sqrt(zTrue[i,j])
        randNxN = np.random.randn(self.numRob, self.numRob) # standard normal distribution.
        np.fill_diagonal(randNxN, 0) # self distance is zero
        zNois = zTrue + randNxN * self.devObser # add noise 正态分布？？
        rand3xN = np.random.randn(self.dimension, self.numRob)
        uNois = u + rand3xN * self.devInput # add noise
        return xTrue, zNois, uNois
