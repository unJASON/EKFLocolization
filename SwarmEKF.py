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
        self.Pmatrix = np.zeros((self.numRob, dimension, dimension))
        self.dt = dt
        self.rd_cnt = 2
        self.rdrange = [[0, [0]*self.rd_cnt]]* numRob
    def EKF(self, uNois, zNois, Esti,xTrue, ekfStride, droneID):
        # 应该要改动
        if self.dimension == 2:
            Q = np.diag([self.Qxy, self.Qxy]) ** 2
        else:
            Q = np.diag([self.Qxy, self.Qxy, self.Qr]) ** 2
        # R = np.diag([self.Rd, self.Rd, self.Rd, self.Rd])**2  # observation covariance
        R = np.diag([self.Rd] * (Esti.shape[1] - 1)) ** 2
        dtEKF = ekfStride * self.dt

        dotXij = np.array(uNois[:, droneID])

        statPred = Esti[:, droneID] + dotXij * dtEKF  # 公式5（1）

        # jacoF = np.array([[1, dtEKF],
        #                   [0, 1]])
        jacoF = np.diag([1.0]*self.dimension)

        # jacoB应该不用了
        PPred = jacoF @ self.Pmatrix[droneID, :, :] @ jacoF.T + dtEKF * dtEKF * Q  # 公式5 （2）

        zPred = []
        jacoH = []
        for i in [jj for jj in range(self.numRob)]:
            if i != droneID:
                distance_temp = 0
                for k in range(self.dimension):
                    distance_temp = distance_temp + (statPred[k] - Esti[k,i])**2
                zPred.insert(i, np.sqrt(distance_temp))
            else:
                zPred.insert(i, 0)

        for i in [jj for jj in range(self.numRob)]:
            if i != droneID:
                dev_temp = []
                for k in range(self.dimension):
                    dev_temp.append((statPred[k]-Esti[k, i])/ zPred[i])
                jacoH.insert(i, dev_temp)
            else:
                if self.dimension == 2:
                    jacoH.insert(i, [0, 0])
                else:
                    jacoH.insert(i,[0,0,0])
        del zPred[droneID]
        del jacoH[droneID]
        temp = zNois[droneID, :].tolist()
        del temp[droneID]

        #adding lighthouse observation
        if droneID in self.lighthouse_Idx:
            if self.dimension == 2:
                zPred.append(statPred[0])
                zPred.append(statPred[1])
                jacoH.append([1,0])
                jacoH.append([0,1])
                temp.append(np.random.randn()*self.lighthouseNoise + xTrue[0,droneID])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[1, droneID])
            else:
                zPred.append(statPred[0])
                zPred.append(statPred[1])
                zPred.append(statPred[2])
                jacoH.append([1,0,0])
                jacoH.append([0,1,0])
                jacoH.append([0,0,1])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[0, droneID])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[1, droneID])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[2, droneID])
            R = np.diag([self.Rd] * R.shape[1] + [self.lighthouseNoise]*self.dimension) ** 2
        jacoH = np.array(jacoH)

        resErr = np.array(temp) - zPred
        S = jacoH @ PPred @ jacoH.T + R
        K = PPred @ jacoH.T @ np.linalg.inv(S)
        statPred = statPred.tolist()
        K = np.array(K)
        Esti[:, droneID] = statPred[:] + K @ resErr  # 公式9 （2）
        self.Pmatrix[droneID, :, :] = (np.eye(len(statPred)) - K @ jacoH) @ PPred  # 公式9 （3）
        return Esti

    def dualEKF(self, uNois, zNois, Esti,xTrue, ekfStride, droneID):
        # 应该要改动
        if self.dimension == 2:
            Q = np.diag([self.Qxy, self.Qxy]) ** 2
        else:
            Q = np.diag([self.Qxy, self.Qxy,self.Qr]) ** 2
        # R = np.diag([self.Rd, self.Rd, self.Rd, self.Rd])**2  # observation covariance
        R = np.diag([self.Rd] * droneID) ** 2
        dtEKF = ekfStride * 0.01

        dotXij = np.array(uNois[:, droneID])

        statPred = Esti[:, droneID] + dotXij * dtEKF  # 公式5（1）

        # jacoF = np.array([[1, dtEKF],
        #                   [0, 1]])
        jacoF = np.diag([1.0]*self.dimension)

        # jacoB应该不用了
        PPred = jacoF @ self.Pmatrix[droneID, :, :] @ jacoF.T + dtEKF * dtEKF * Q  # 公式5 （2）

        zPred = []
        jacoH = []
        for i in [jj for jj in range(droneID+1)]:
            if i != droneID:
                distance_temp = 0
                for k in range(self.dimension):
                    distance_temp = distance_temp + (statPred[k] - Esti[k,i])**2
                zPred.insert(i, np.sqrt(distance_temp))
            else:
                zPred.insert(i, 0)

        for i in [jj for jj in range(droneID+1)]:
            if i != droneID:
                dev_temp = []
                for k in range(self.dimension):
                    dev_temp.append((statPred[k]-Esti[k, i])/ zPred[i])
                jacoH.insert(i, dev_temp)
            else:
                if self.dimension == 2:
                    jacoH.insert(i, [0, 0])
                else:
                    jacoH.insert(i,[0,0,0])
        del zPred[droneID]
        del jacoH[droneID]
        temp = zNois[droneID, :].tolist()
        temp = temp[:droneID + 1]
        del temp[droneID]

        #adding lighthouse observation
        if droneID in self.lighthouse_Idx:
            if self.dimension == 2:
                zPred.append(statPred[0])
                zPred.append(statPred[1])
                jacoH.append([1,0])
                jacoH.append([0,1])
                temp.append(np.random.randn()*self.lighthouseNoise + xTrue[0,droneID])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[1, droneID])
            else:
                zPred.append(statPred[0])
                zPred.append(statPred[1])
                zPred.append(statPred[2])
                jacoH.append([1,0,0])
                jacoH.append([0,1,0])
                jacoH.append([0,0,1])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[0, droneID])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[1, droneID])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[2, droneID])
            R = np.diag([self.Rd]* R.shape[1] + [self.lighthouseNoise]*self.dimension) ** 2
        jacoH = np.array(jacoH)

        resErr = np.array(temp) - zPred
        S = jacoH @ PPred @ jacoH.T + R
        K = PPred @ jacoH.T @ np.linalg.inv(S)
        statPred = statPred.tolist()
        K = np.array(K)
        Esti[:, droneID] = statPred + K @ resErr  # 公式9 （2）
        self.Pmatrix[droneID, :, :] = (np.eye(len(statPred)) - K @ jacoH) @ PPred  # 公式9 （3）

        return Esti

    #only range with anchor
    def anchorEKF(self, uNois, zNois, Esti,xTrue, ekfStride, droneID):
        # 应该要改动
        if self.dimension == 2:
            Q = np.diag([self.Qxy, self.Qxy]) ** 2
        else:
            Q = np.diag([self.Qxy, self.Qxy, self.Qr]) ** 2
        # R = np.diag([self.Rd, self.Rd, self.Rd, self.Rd])**2  # observation covariance

        dtEKF = ekfStride * 0.01

        dotXij = np.array(uNois[:, droneID])

        statPred = Esti[:, droneID] + dotXij * dtEKF  # 公式5（1）

        # jacoF = np.array([[1, dtEKF],
        #                   [0, 1]])
        jacoF = np.diag([1.0]*self.dimension)

        # jacoB应该不用了
        PPred = jacoF @ self.Pmatrix[droneID, :, :] @ jacoF.T + dtEKF * dtEKF * Q  # 公式5 （2）

        zPred = []
        jacoH = []
        for i in [jj for jj in range(self.numRob)]:
            if i != droneID and i in self.lighthouse_Idx:
                distance_temp = 0
                for k in range(self.dimension):
                    distance_temp = distance_temp + (statPred[k] - Esti[k,i])**2
                zPred.append(np.sqrt(distance_temp))
                dev_temp = []
                for k in range(self.dimension):
                    dev_temp.append((statPred[k] - Esti[k, i]) / np.sqrt(distance_temp))
                jacoH.append(dev_temp)
            else:
                pass



        temp = zNois[droneID, :].tolist()
        temp = [ele for (idx,ele) in enumerate(temp) if idx!= droneID and idx in self.lighthouse_Idx]
        R_cnt = len(temp)
        R = np.diag([self.Rd] * R_cnt) ** 2
        #adding lighthouse observation
        if droneID in self.lighthouse_Idx:
            if self.dimension == 2:
                zPred.append(statPred[0])
                zPred.append(statPred[1])
                jacoH.append([1,0])
                jacoH.append([0,1])
                temp.append(np.random.randn()*self.lighthouseNoise + xTrue[0,droneID])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[1, droneID])
            else:
                zPred.append(statPred[0])
                zPred.append(statPred[1])
                zPred.append(statPred[2])
                jacoH.append([1,0,0])
                jacoH.append([0,1,0])
                jacoH.append([0,0,1])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[0, droneID])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[1, droneID])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[2, droneID])
            R = np.diag([self.Rd] * R.shape[1] + [self.lighthouseNoise]*self.dimension) ** 2
        jacoH = np.array(jacoH)

        resErr = np.array(temp) - zPred
        S = jacoH @ PPred @ jacoH.T + R
        K = PPred @ jacoH.T @ np.linalg.inv(S)
        statPred = statPred.tolist()
        K = np.array(K)
        Esti[:, droneID] = statPred[:] +  K @ resErr  # 公式9 （2）
        self.Pmatrix[droneID, :, :] = (np.eye(len(statPred)) - K @ jacoH) @ PPred  # 公式9 （3）
        return Esti
    # choose 2 points (2 random)
    def rdEKF(self, uNois, zNois, Esti,xTrue, ekfStride, droneID):

        Q = np.diag([self.Qxy, self.Qxy]) ** 2
        # R = np.diag([self.Rd, self.Rd, self.Rd, self.Rd])**2  # observation covariance
        R = np.diag([self.Rd] * self.rd_cnt) ** 2
        dtEKF = ekfStride * 0.01

        dotXij = np.array(uNois[:, droneID])

        statPred = Esti[:, droneID] + dotXij * dtEKF  # 公式5（1）

        # jacoF = np.array([[1, dtEKF],
        #                   [0, 1]])
        jacoF = np.diag([1.0] * self.dimension)

        # jacoB应该不用了
        PPred = jacoF @ self.Pmatrix[droneID, :, :] @ jacoF.T + dtEKF * dtEKF * Q  # 公式5 （2）

        zPred = []
        jacoH = []
        for i in [jj for jj in range(self.numRob)]:
            if i != droneID:
                distance_temp = 0
                for k in range(self.dimension):
                    distance_temp = distance_temp + (statPred[k] - Esti[k, i]) ** 2
                zPred.insert(i, np.sqrt(distance_temp))
            else:
                zPred.insert(i, 0)

        for i in [jj for jj in range(self.numRob)]:
            if i != droneID:
                dev_temp = []
                for k in range(self.dimension):
                    dev_temp.append((statPred[k] - Esti[k, i]) / zPred[i])
                jacoH.insert(i, dev_temp)
            else:
                if self.dimension == 2:
                    jacoH.insert(i, [0, 0])
                else:
                    jacoH.insert(i, [0, 0, 0])
        del zPred[droneID]
        del jacoH[droneID]
        temp = zNois[droneID, :].tolist()
        del temp[droneID]

        rand_choice = self.rdrange[droneID][1]
        if self.rdrange[droneID][0] == 0:

            rand_choice = np.random.choice(range(1,len(zPred)), self.rd_cnt-1, replace=False).tolist()
            rand_choice.append(0)
            # rand_choice.append(np.random.randint(1,len(zPred)))
            self.rdrange[droneID]=[500,rand_choice]
        else:
            self.rdrange[droneID][0] = self.rdrange[droneID][0]-1

        zPred = [ele for (idx,ele) in enumerate(zPred) if idx in rand_choice]
        jacoH = [ele for (idx, ele) in enumerate(jacoH) if idx in rand_choice]
        temp = [ele for (idx, ele) in enumerate(temp) if idx in rand_choice]


        # adding lighthouse observation
        if droneID in self.lighthouse_Idx:
            if self.dimension == 2:
                zPred.append(statPred[0])
                zPred.append(statPred[1])
                jacoH.append([1, 0])
                jacoH.append([0, 1])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[0, droneID])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[1, droneID])
            else:
                zPred.append(statPred[0])
                zPred.append(statPred[1])
                zPred.append(statPred[2])
                jacoH.append([1, 0, 0])
                jacoH.append([0, 1, 0])
                jacoH.append([0, 0, 1])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[0, droneID])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[1, droneID])
                temp.append(np.random.randn() * self.lighthouseNoise + xTrue[2, droneID])
            R = np.diag([self.Rd] * R.shape[1] + [self.lighthouseNoise] * self.dimension) ** 2
        jacoH = np.array(jacoH)

        resErr = np.array(temp) - zPred
        S = jacoH @ PPred @ jacoH.T + R
        K = PPred @ jacoH.T @ np.linalg.inv(S)
        statPred = statPred.tolist()
        K = np.array(K)
        Esti[:, droneID] = statPred[:] + K @ resErr  # 公式9 （2）
        self.Pmatrix[droneID, :, :] = (np.eye(len(statPred)) - K @ jacoH) @ PPred  # 公式9 （3）

        return Esti

    #CovEKF
    def CovEKF(self, uNois, zNois, Esti,xTrue, ekfStride, droneID):
        # 应该要改动
        if self.dimension == 2:
            Q = np.diag([self.Qxy, self.Qxy]) ** 2
        else:
            Q = np.diag([self.Qxy, self.Qxy, self.Qr]) ** 2
        dtEKF = ekfStride * self.dt
        dotXij = np.array(uNois[:, droneID])
        statPred = Esti[:, droneID] + dotXij * dtEKF  # 公式5（1）
        jacoF = np.diag([1.0]*self.dimension)
        # jacoB应该不用了
        PPred = jacoF @ self.Pmatrix[droneID, :, :] @ jacoF.T + dtEKF * dtEKF * Q  # 公式5 （2）

        #adding light house_gain
        statPred,PPred = self.lighthouse_Gain(droneID, statPred, PPred, xTrue)
        if droneID not in self.lighthouse_Idx:
        # if True:
            # add drone gain
            # choice_list =np.random.choice(range(1, self.numRob), np.random.randint(2,3),replace=False).tolist()
            # for i in choice_list:
            for i in range(self.numRob):
                # if i != droneID:
                if i != droneID and i in self.lighthouse_Idx:
                    statPred, PPred = self.drone_Gain(droneID, statPred, PPred, i, Esti, zNois)
                else:
                    pass
        # for i in range(np.random.randint(1,self.numRob//2)):
        Esti[:, droneID] = statPred[:]
        self.Pmatrix[droneID, :, :] = PPred
        return Esti

    def drone_Gain(self,droneID,statPred,PPred,reference_Drone,Esti,zNois):
        #noise
        R = np.diag([self.Rd] * statPred.shape[0]*2)**2

        #reference
        stat_reference = Esti[:,reference_Drone]
        P_reference = self.Pmatrix[reference_Drone, :, :]

        PPred1 = np.hstack((PPred,np.zeros([self.dimension,self.dimension])))
        P_reference1 = np.hstack((np.zeros([self.dimension,self.dimension]),P_reference))
        combine_P = np.vstack([PPred1,P_reference1])
        combine_stat = np.hstack([statPred,stat_reference])

        Z_observation = zNois[droneID, reference_Drone]
        jacoH = []
        zPred = 0
        for k in range(self.dimension):
            zPred = zPred + (statPred[k] - stat_reference[k]) ** 2
        zPred = np.sqrt(zPred)
        for k in range(self.dimension):
            jacoH.append( (statPred[k] - stat_reference[k]) / zPred )
        for k in range(self.dimension):
            jacoH.append( (stat_reference[k]-statPred[k] ) / zPred )


        jacoH = np.array(jacoH)
        resErr = np.array(np.array(Z_observation) - zPred).reshape([1,])

        S = jacoH @ combine_P @ jacoH.T + R
        K = combine_P @ jacoH.T @ np.linalg.inv(S)
        K = K.reshape([K.shape[0], 1])
        combine_stat=combine_stat.tolist()
        if reference_Drone in self.lighthouse_Idx:
            combine_stat = combine_stat[:] + K @ resErr  # 公式9 （2）
        else:
            combine_stat = combine_stat[:] + K @ resErr  # 公式9 （2）
        jacoH = jacoH.reshape([1,jacoH.shape[0]])
        combine_P = (np.eye(len(combine_stat)) - K @ jacoH) @ combine_P
        return combine_stat[:self.dimension],combine_P[:self.dimension,:self.dimension]

    def lighthouse_Gain(self,droneID,statPred,PPred,xTrue):
        if droneID in self.lighthouse_Idx:
            R = np.diag([self.lighthouseNoise]*self.dimension)**2
            Z_observation = []
            jacoH = []
            if self.dimension == 2:
                jacoH.append([1, 0])
                jacoH.append([0, 1])
                Z_observation.append(np.random.randn()*self.lighthouseNoise + xTrue[0,droneID])
                Z_observation.append(np.random.randn()*self.lighthouseNoise + xTrue[1,droneID])
            else:
                jacoH.append([1, 0, 0])
                jacoH.append([0, 1, 0])
                jacoH.append([0, 0, 1])
                Z_observation.append(np.random.randn() * self.lighthouseNoise + xTrue[0, droneID])
                Z_observation.append(np.random.randn() * self.lighthouseNoise + xTrue[1, droneID])
                Z_observation.append(np.random.randn() * self.lighthouseNoise + xTrue[2, droneID])

            jacoH = np.array(jacoH)
            resErr = np.array(Z_observation) - statPred
            S = jacoH @ PPred @ jacoH.T + R
            K = PPred @ jacoH.T @ np.linalg.inv(S)
            statPred = statPred.tolist()
            K = np.array(K)
            statPred = statPred[:] + K @ resErr  # 公式9 （2）
            PPred = (np.eye(len(statPred)) - K @ jacoH) @ PPred
        else:
            pass
        return statPred,PPred
