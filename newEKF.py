
import numpy as np

class newEKF :
    def __init__(self, Pxy, Pr, Qxy, Qr, Rd, numRob):
        self.Pxy = Pxy
        self.Pr = Pr
        self.Qxy = Qxy
        self.Qr = Qr
        self.Rd = Rd
        self.numRob = numRob
        self.Pmatrix = np.zeros((self.numRob+1, 2, 2))
        #self.Pmatrix = np.zeros((3, 3, self.numRob, self.numRob))
        #for i in range(self.numRob):
            #for j in range(self.numRob):
                #self.Pmatrix[0:2, 0:2, i, j] = np.eye(2)*Pxy
                #self.Pmatrix[2, 2, i, j] = Pr

    def nEKF(self, uNois, zNois, Esti, ekfStride, droneID):
        #应该要改动
        Q = np.diag([self.Qxy, self.Qxy])**2
        # R = np.diag([self.Rd, self.Rd, self.Rd, self.Rd])**2  # observation covariance
        R = np.diag([self.Rd]*(Esti.shape[1]-1)) ** 2
        dtEKF = ekfStride*0.01
        uVix, uViy, uRi = uNois[:, droneID]
        dotXij = np.array([uVix, uViy, uRi])

        statPred = Esti[:, droneID] + dotXij * dtEKF #公式5（1）

        # jacoF=np.array([[1, dtEKF],
        #                 [0, 1]])
        jacoF = np.array([[1, 0],
                          [0, 1]])
        #jacoB应该不用了

        PPred = jacoF@self.Pmatrix[droneID, :, :]@jacoF.T + dtEKF*dtEKF*Q#公式5 （2）

        xij, yij, yawij = statPred
 
        zPred = []
        jacoH = []
        for i in [jj for jj in range(self.numRob)]:
            if i != droneID:
                zPred.insert(i, np.sqrt((xij - Esti[0, i])**2 + (yij - Esti[1, i])**2))
            else:
                zPred.insert(i,0)

        for i in [jj for jj in range(self.numRob)]:
            if i != droneID:
                xi = (xij-Esti[0, i])/zPred[i]
                xj = (yij-Esti[1, i])/zPred[i]
                jacoH.insert(i, [xi, xj])
            else:
                jacoH.insert(i,[0,0])
        del zPred[droneID]
        del jacoH[droneID]
        temp = zNois[droneID, :].tolist()
        del temp[droneID]

        jacoH = np.array(jacoH)


        resErr = np.array(temp) - zPred
        S = jacoH@PPred@jacoH.T + R
        K = PPred@jacoH.T@np.linalg.inv(S)
        statPred = statPred.tolist()       
        K = np.array(K)
        Esti[0:2, droneID] = statPred[0:2]+ K@resErr#公式9 （2）
        self.Pmatrix[droneID, :, :] = (np.eye(len(statPred[0:2])) - K@jacoH)@PPred#公式9 （3）
        # self.Pmatrix[droneID,0,1] = 0
        # self.Pmatrix[droneID,1, 0] = 0
        # print(self.Pmatrix[1, 0, 1])
        return Esti

    def dualEKF(self, uNois, zNois, Esti, ekfStride, droneID):
        # 应该要改动
        Q = np.diag([self.Qxy, self.Qxy]) ** 2
        # R = np.diag([self.Rd, self.Rd, self.Rd, self.Rd])**2  # observation covariance
        # R = np.diag([self.Rd, self.Rd,self.Rd]) ** 2
        R = np.diag([self.Rd]*droneID)**2
        dtEKF = ekfStride * 0.01

        uVix, uViy, uRi = uNois[:, droneID]
        dotXij = np.array([uVix, uViy, uRi])

        statPred = Esti[:, droneID] + dotXij * dtEKF  # 公式5（1）
        # jacoF = np.array([[1, dtEKF],
        #                   [0, 1]])
        jacoF=np.array([[1, 0],
                    [0, 1]])
        # jacoB应该不用了
        PPred = jacoF @ self.Pmatrix[droneID, :, :] @ jacoF.T + dtEKF*dtEKF*Q# 公式5 （2）
        xij, yij, yawij = statPred

        zPred = []
        jacoH = []
        for i in [jj for jj in range(droneID+1)]:
            if i != droneID:
                zPred.insert(i, np.sqrt((xij - Esti[0, i]) ** 2 + (yij - Esti[1, i]) ** 2))
            else:
                zPred.insert(i, 0)

        for i in [jj for jj in range(droneID+1)]:
            if i != droneID:
                xi = (xij - Esti[0, i]) / zPred[i]
                xj = (yij - Esti[1, i]) / zPred[i]
                jacoH.insert(i, [xi, xj])
            else:
                jacoH.insert(i, [0, 0])
        del zPred[droneID]
        del jacoH[droneID]
        temp = zNois[droneID, :].tolist()
        temp = temp[:droneID+1]
        del temp[droneID]

        jacoH = np.array(jacoH)

        resErr = np.array(temp) - zPred
        S = jacoH @ PPred @ jacoH.T + R
        K = PPred @ jacoH.T @ np.linalg.inv(S)
        statPred = statPred.tolist()
        K = np.array(K)
        Esti[0:2, droneID] = statPred[0:2] + K @ resErr  # 公式9 （2）
        self.Pmatrix[droneID, :, :] = (np.eye(len(statPred[0:2])) - K @ jacoH) @ PPred  # 公式9 （3）
        return Esti



    def singleEKF(self, uNois, zNois, Esti, ekfStride, droneID):
        # 应该要改动
        Q = np.diag([self.Qxy, self.Qxy]) ** 2
        # R = np.diag([self.Rd, self.Rd, self.Rd, self.Rd])**2  # observation covariance
        # R = np.diag([self.Rd, self.Rd,self.Rd]) ** 2
        R = np.diag([self.Rd])**2
        dtEKF = ekfStride * 0.01

        uVix, uViy, uRi = uNois[:, droneID]
        dotXij = np.array([uVix, uViy, uRi])

        statPred = Esti[:, droneID] + dotXij * dtEKF  # 公式5（1）
        jacoF = np.array([[1, dtEKF],
                          [0, 1]])
        # jacoB应该不用了
        PPred = jacoF @ self.Pmatrix[droneID, :, :] @ jacoF.T + dtEKF*dtEKF*Q# 公式5 （2）
        xij, yij, yawij = statPred

        zPred = []
        jacoH = []
        for i in [jj for jj in range(1)]:
            zPred.insert(i, np.sqrt((xij - Esti[0, i]) ** 2 + (yij - Esti[1, i]) ** 2))

        for i in [jj for jj in range(1)]:
            xi = (xij - Esti[0, i]) / zPred[i]
            xj = (yij - Esti[1, i]) / zPred[i]
            jacoH.insert(i, [xi, xj])

        jacoH = np.array(jacoH)

        resErr = zNois[droneID, 1] - zPred
        S = jacoH @ PPred @ jacoH.T + R
        K = PPred @ jacoH.T @ np.linalg.inv(S)
        statPred = statPred.tolist()
        K = np.array(K)
        Esti[0:2, droneID] = statPred[0:2] + K @ resErr  # 公式9 （2）
        self.Pmatrix[droneID, :, :] = (np.eye(len(statPred[0:2])) - K @ jacoH) @ PPred  # 公式9 （3）
        # print(self.Pmatrix[1, 0, :])

        return Esti

