import numpy as np



class swarmEKF:
    def __init__(self, Pxy, Pr, Qxy, Qr, Rd, numRob,dt=0.01,dimension=2,lighthouse_Idx=[0],lighthouseNoise=0.01):
        self.Pxy = Pxy
        self.Pr = Pr
        self.Qxy = Qxy
        self.Qr = Qr
        self.Rd = Rd
        self.numRob = numRob
        self.dimension = dimension
        self.lighthouse_Idx=lighthouse_Idx
        self.lighthouseNoise=lighthouseNoise
        self.dt = dt
        self.Pmatrix = np.zeros((numRob, dimension * 2, dimension * 2))
    #CovEKF
    def CovEKF(self, uNois, zNois, Esti,xTrue, ekfStride, droneID):
        # 应该要改动
        if self.dimension == 2:
            Q = np.diag([0.005,0.005,self.Qxy, self.Qxy]) ** 2
        else:
            Q = np.diag([self.Qxy, self.Qxy, self.Qr]) ** 2
        dtEKF = ekfStride * self.dt
        dotXij = np.array([0,0]+uNois[:, droneID].tolist())
        statPred = Esti[:, droneID] + dotXij * dtEKF  # 公式5（1）
        jacoF = np.diag([1.0]*self.dimension*2)
        # jacoB应该不用了
        Q[2:,:] = dtEKF * dtEKF *Q[2:,:]
        PPred = jacoF @ self.Pmatrix[droneID, :, :] @ jacoF.T + Q  # 公式5 （2）
        #adding velocity gain
        # statPred, PPred = self.velocity_Gain(droneID,statPred, PPred,Esti,uNois[:, droneID],dtEKF)

        #adding light house_gain
        statPred,PPred = self.lighthouse_Gain(droneID, statPred, PPred, xTrue)

        #adding drone_gain
        if droneID not in self.lighthouse_Idx:
            # choice_list =np.random.choice(range(1, self.numRob), np.random.randint(2,3),replace=False).tolist()
            # for i in choice_list:
            # for i in range(self.numRob):
            for i in range(droneID):
                if i != droneID and i in self.lighthouse_Idx:
                    # print(statPred,PPred)
                    statPred, PPred = self.drone_Gain(droneID, statPred, PPred, i, Esti, zNois)
                    # if np.abs(statPred[0]) > 100 or np.abs(PPred[0][0]) > 100:
                    #     print("hello bug")

                else:
                    pass
        # for i in range(np.random.randint(1,self.numRob//2)):
        #height

        #anglo
        Esti[:, droneID] = statPred[:]

        self.Pmatrix[droneID, :, :] = PPred
        return Esti

    def CovEKF2(self, uNois, zNois, Esti,xTrue, ekfStride, droneID,u):
        # 应该要改动
        if self.dimension == 2:
            Q = np.diag([0.005,0.005,self.Qxy, self.Qxy]) ** 2
        else:
            Q = np.diag([self.Qxy, self.Qxy, self.Qr]) ** 2
        dtEKF = ekfStride * self.dt
        dotXij = np.array([0,0]+uNois[:, droneID].tolist())
        statPred = Esti[:, droneID] + dotXij * dtEKF  # 公式5（1）
        jacoF = np.diag([1.0]*self.dimension*2)
        # jacoB应该不用了
        Q[2:,:] = dtEKF * dtEKF *Q[2:,:]
        PPred = jacoF @ self.Pmatrix[droneID, :, :] @ jacoF.T + Q  # 公式5 （2）

        #adding velocity gain
        statPred, PPred = self.velocity_Gain(droneID,statPred, PPred,Esti,uNois[:, droneID],dtEKF,u[:,droneID])

        #adding light house_gain
        statPred,PPred = self.lighthouse_Gain(droneID, statPred, PPred, xTrue)

        #adding drone_gain
        if droneID not in self.lighthouse_Idx:
            # choice_list =np.random.choice(range(1, self.numRob), np.random.randint(2,3),replace=False).tolist()
            # for i in choice_list:
            # for i in range(self.numRob):
            for i in range(droneID):
                if i != droneID and i in self.lighthouse_Idx:
                    statPred, PPred = self.drone_Gain(droneID, statPred, PPred, i, Esti, zNois)
                else:
                    pass
        # for i in range(np.random.randint(1,self.numRob//2)):
        #height
        #anglo
        Esti[:, droneID] = statPred[:]

        self.Pmatrix[droneID, :, :] = PPred
        return Esti

    def drone_Gain(self,droneID,statPred,PPred,reference_Drone,Esti,zNois):
        #noise
        R = np.diag([self.Rd] * statPred.shape[0]*2)**2

        #reference
        stat_reference = Esti[:,reference_Drone]
        P_reference = self.Pmatrix[reference_Drone, :, :]

        PPred1 = np.hstack((PPred,np.zeros([self.dimension*2,self.dimension*2])))
        P_reference1 = np.hstack((np.zeros([self.dimension*2,self.dimension*2]),P_reference))
        combine_P = np.vstack([PPred1,P_reference1])
        combine_stat = np.hstack([statPred,stat_reference])

        Z_observation = zNois[droneID, reference_Drone]
        jacoH = []
        zPred = 0
        for k in range(self.dimension):
            zPred = zPred + (statPred[k]+statPred[k+self.dimension] - stat_reference[k]- stat_reference[k+self.dimension]) ** 2
        zPred = np.sqrt(zPred)
        # jacoH_1,jacoH_2 = [],[]
        # for k in range(self.dimension):
        #     jacoH_1.append( (statPred[k] + statPred[k+2] - stat_reference[k] - stat_reference[k+2]) / zPred )
        # jacoH_1 = jacoH_1 + jacoH_1
        # for k in range(self.dimension):
        #     jacoH_2.append( (stat_reference[k]+stat_reference[k+2]-statPred[k] - statPred[k+2] ) / zPred )
        # jacoH_2 = jacoH_2 + jacoH_2
        # jacoH = jacoH_1 + jacoH_2
        jacoH.append(( (statPred[0] + statPred[2] - stat_reference[0] - stat_reference[2]) / zPred ))
        jacoH.append(( (statPred[1] + statPred[3] - stat_reference[1] - stat_reference[3]) / zPred ))
        jacoH.append(((statPred[0] + statPred[2] - stat_reference[0] - stat_reference[2]) / zPred  ))
        jacoH.append(((statPred[1] + statPred[3] - stat_reference[1] - stat_reference[3]) / zPred  ))

        jacoH.append(((stat_reference[0] + stat_reference[2] - statPred[0] - statPred[2]) / zPred))
        jacoH.append(((stat_reference[1] + stat_reference[3] - statPred[1] - statPred[3]) / zPred))
        jacoH.append(((stat_reference[0] + stat_reference[2] - statPred[0] - statPred[2]) / zPred))
        jacoH.append(((stat_reference[1] + stat_reference[3] - statPred[1] - statPred[3]) / zPred))

        jacoH = np.array(jacoH)
        resErr = np.array(np.array(Z_observation) - zPred).reshape([1,])

        S = jacoH @ combine_P @ jacoH.T + R
        try:
            K = combine_P @ jacoH.T @ np.linalg.inv(S)
        except:
            print(S)
        else:
            pass
        K = K.reshape([K.shape[0], 1])
        combine_stat=combine_stat.tolist()
        combine_stat = combine_stat[:] + K @ resErr  # 公式9 （2）
        jacoH = jacoH.reshape([1,jacoH.shape[0]])
        combine_P = (np.eye(len(combine_stat)) - K @ jacoH) @ combine_P
        return combine_stat[:self.dimension*2],combine_P[:self.dimension*2,:self.dimension*2]

    def lighthouse_Gain(self,droneID,statPred,PPred,xTrue):
        if droneID in self.lighthouse_Idx:
            R = np.diag([self.lighthouseNoise]*self.dimension)**2
            Z_observation = []
            jacoH = []
            
            if self.dimension == 2:
                jacoH.append([1, 0,1,0])
                jacoH.append([0, 1,0,1])
                Z_observation.append(np.random.randn()*self.lighthouseNoise + xTrue[0,droneID])
                Z_observation.append(np.random.randn()*self.lighthouseNoise + xTrue[1,droneID])
                h_state_pred = [statPred[0]+ statPred[2],statPred[1]+statPred[3]]
            else:
                jacoH.append([1, 0, 0,1,0,0])
                jacoH.append([0, 1, 0,0, 1, 0])
                jacoH.append([0, 0, 1,0, 0, 1])
                Z_observation.append(np.random.randn() * self.lighthouseNoise + xTrue[0, droneID])
                Z_observation.append(np.random.randn() * self.lighthouseNoise + xTrue[1, droneID])
                Z_observation.append(np.random.randn() * self.lighthouseNoise + xTrue[2, droneID])
                h_state_pred = [statPred[0] + statPred[0+self.dimension], statPred[1] + statPred[1+self.dimension],statPred[2] + statPred[2+self.dimension]]
            jacoH = np.array(jacoH)
            resErr = np.array(Z_observation) - h_state_pred
            S = jacoH @ PPred @ jacoH.T + R
            K = PPred @ jacoH.T @ np.linalg.inv(S)
            statPred = statPred.tolist()
            K = np.array(K)
            statPred = statPred[:] + K @ resErr  # 公式9 （2）
            PPred = (np.eye(len(statPred)) - K @ jacoH) @ PPred
        else:
            pass
        return statPred,PPred

    def velocity_Gain(self, droneID, statPred, PPred, Esti,uNoise,dtEKF,u):
        R = np.diag([self.Qxy, self.Qxy]) ** 2
        R = dtEKF*dtEKF*R
        Z_observation = []
        jacoH = []
        if self.dimension == 2:
            jacoH.append([0, 0, 1, 0])
            jacoH.append([0, 0, 0, 1])
            Z_observation.append(u[0]*dtEKF+dtEKF*np.random.randn()*self.Qxy)
            Z_observation.append(u[1]*dtEKF+dtEKF*np.random.randn()*self.Qxy)
            h_state_pred = [statPred[2] - Esti[2,droneID], statPred[3] - Esti[3,droneID]]
        else:
            jacoH.append([0, 0, 0, 1, 0, 0])
            jacoH.append([0, 0, 0, 0, 1, 0])
            jacoH.append([0, 0, 0, 0, 0, 1])
            Z_observation.append(uNoise[0] * dtEKF)
            Z_observation.append(uNoise[1] * dtEKF)
            Z_observation.append(uNoise[2] * dtEKF)
            h_state_pred = [statPred[3] - Esti[3,droneID],
                            statPred[4] - Esti[4,droneID],
                            statPred[5] - Esti[5,droneID]]

        jacoH = np.array(jacoH)
        resErr = np.array(Z_observation) - h_state_pred
        S = jacoH @ PPred @ jacoH.T + R
        K = PPred @ jacoH.T @ np.linalg.inv(S)
        statPred = statPred.tolist()
        K = np.array(K)
        statPred = statPred[:] + K @ resErr  # 公式9 （2）
        PPred = (np.eye(len(statPred)) - K @ jacoH) @ PPred

        return statPred, PPred