"""
Author: Zhang Qianyi
Mail: 112021019@mail.nankai.edu.cn
Latest update: 2021.12.9
"""

#TODO:ppt中的G有错误，不同行的错列不是同一个值。
#TODO:ppt中的phy(k)有错误，deltaU应该到k-nb-1，不然theta和phy的维度不一样
#reference：https://zhuanlan.zhihu.com/p/368705959
from tkinter.constants import NO
import numpy as np
import matplotlib.pyplot as plt
import random


class GPC:
    def __init__(self, theta0, beta, N, Nu, lamda, alpha, init_u, init_y, na, nb):
        self.theta0 = theta0    # 反馈矫正
        self.beta = beta    # 反馈矫正
        self.N = N  # 预测视野
        self.Nu = Nu    # 控制视野
        self.lamda = lamda  # 能量代价的权重
        self.alpha = alpha  # 柔化因子
        self.init_u = init_u    # 历史输入=[-1时刻，-2时刻]
        self.init_y = init_y    # 历史输出=[-1时刻，-2时刻]
        self.na = na    # a的长度
        self.nb = nb    # b的长度
        self.theta_k = [] # 无模型初始化


    def setTargetTraj(self, target_state=10):
        """
        给定目标信号
        """
        self.soft_traj = []
        for i in range(0,self.N):
            if i==0:
                self.soft_traj.append(self.alpha*self.init_y[0] + (1-self.alpha)*target_state)
            else:
                self.soft_traj.append(self.alpha*self.soft_traj[i-1] + (1-self.alpha)*target_state)


    def setModel(self, A=[1,-1.5,0.7], B=[1,0.5]):
        """
        有模型控制
        """
        self.A = A
        self.B = B


    def calModel(self, his_y, his_u):
        """
        无模型控制
        """
        if len(his_y)>=self.na+1 and len(his_u)>=self.nb+1:
            # 有足够多的历史数据时，才可以进行反馈矫正，否则返回error
            if len(self.theta_k)==0:
                # 初始化A和B
                self.theta_k = np.array(self.theta0).T
                self.p_k = np.array(self.beta*self.beta*np.eye(len(self.theta_k)))
                self.A = [1.0]
                self.B = []
                for i,value in enumerate(self.theta_k):
                    if i<self.na:
                        self.A.append(-value)
                    else:
                        self.B.append(value)
            else:
                # 更新A和B
                phy = []
                for i in range(self.na):
                    phy.append(his_y[-i-1]-his_y[-i-2])
                for i in range(self.nb):
                    phy.append(his_u[-i-1]-his_u[-i-2])
                phy = np.array(phy).T
                self.theta_k += (self.Yk-his_y[-1] - np.dot(self.theta_k.T,phy)) / (1+np.dot(np.dot(phy.T,self.p_k),phy)) * np.dot(self.p_k,phy)    # 注意这里用的Yk是真实值（可观测）
                self.p_k -= np.dot(np.dot(np.dot(self.p_k,phy),phy.T),self.p_k) /(1+np.dot(np.dot(phy.T,self.p_k),phy))
                self.A = [1.0]
                self.B = []
                for i,value in enumerate(self.theta_k):
                    if i<self.na:
                        self.A.append(-value)
                    else:
                        self.B.append(value)
            print('A:',self.A)
            print('B:',self.B)
            return True, None
        else:
            return False, 'no sufficient history datas'


    def calU(self):
        """
        计算G和H，进而计算输出控制量
        """
        # A_minus1: e^{-1}*A
        self.A_minus1 = [0]
        for value in self.A:
            self.A_minus1.append(value)
        # A_overline = A*DELTA = A*(1-e^{-1})=A-A_minus
        self.A_overline = [-value for value in self.A_minus1]
        for i in range(len(self.A)):
            self.A_overline[i] += self.A[i]
        # print(self.A_overline)
        
        # E1 and F1
        E = [[1]]
        temp_F = []
        for i,value in enumerate(self.A_overline):
            if i!=0:
                temp_F.append(-value)
        F = [temp_F]
        # print(E)
        # print(F)

        # E_{j+1} and F_{j+1}
        for j in range(0,self.N):
            # temp_E
            e_j=F[j][0]
            # print('-')
            # print(e_j)
            temp_E = E[j].copy()
            temp_E.append(e_j)
            E.append(temp_E)
            # temp_F
            temp_F = [0 for i in range(max(len(F[j]),len(self.A_overline)))]
            for i,value in enumerate(F[j]):
                temp_F[i]+= value
            for i,value in enumerate(self.A_overline):
                temp_F[i] += -value*e_j
            temp_F_shift = []
            for i,value in enumerate(temp_F):
                if i!=0:
                    temp_F_shift.append(value)
            F.append(temp_F_shift)
            # print(E)
            # print(F)

        # Ej_B
        Ej_B=[]
        for j in range(0,self.N):
            temp_Ej_B = [0 for i in range(max(len(E[j])+1,len(self.B)))]
            for i,value in enumerate(E[j]):
                temp_Ej_B[i] += value*self.B[0]
                temp_Ej_B[i+1] += value*self.B[1]
            Ej_B.append(temp_Ej_B)
        # print(Ej_B)

        # G_j & H_j
        G_j = []
        H_j = []
        for j in range(0,self.N):
            G_j=Ej_B[j][:j+1]
            H_j.append(Ej_B[j][j+1])
        # print(G_j)
        # print(H_j)
            
        #y_0
        Y_k_plus_j = []
        for j in range(0,self.N):
            temp_Fj_y = 0
            for i,value in enumerate(F[j]):
                if i==0:
                    temp_Fj_y += value*self.Yk
                else:
                    temp_Fj_y += value*self.init_y[i-1]
            temp_Y_k_plus_j = H_j[j]*(self.init_u[0]-self.init_u[1]) + temp_Fj_y
            Y_k_plus_j.append(temp_Y_k_plus_j)

        #G
        G = np.zeros([self.N,self.Nu])
        for j in range(self.N):
            for i in range(self.Nu):
                if i+j>=self.N:
                    break
                G[j+i][i]=G_j[j]
        G=np.array(G)
        # print(G)

        #combineation
        res = np.linalg.inv(np.dot(G.T,G)+self.lamda*np.array(np.eye(len(G.T))))
        print('-')
        print('soft traj: ',self.init_y[0],self.soft_traj)

        #delta_y      
        delta_y = []
        for i in range(len(self.soft_traj)):
            delta_y.append(self.soft_traj[i]-Y_k_plus_j[i])
        delta_Uk = np.dot(np.dot(res,G.T),np.array(delta_y))
        # print(delta_Uk[0])
        
        Y_prdict = np.array(Y_k_plus_j) + np.dot(G,delta_Uk)
        print('predict traj: ',Y_prdict)
        print('control:', self.init_u[0] + delta_Uk[0])
        return self.init_u[0] + delta_Uk[0]

    
    def setYk(self, Yk):
        self.Yk=Yk
        # print('YK:',self.Yk)





class env():
    def __init__(self, A=[1,-1.5,0.7], B=[1,0.5], init_u=[0,1], init_y=[0,0.2], sigma=0.01):
        """
        初始化仿真环境
        """
        self.A = A  # 模型参数，仅用于有模型控制
        self.B = B  # 模型参数，仅用于有模型控制
        self.u = init_u # k-1和k-2时刻的控制量
        self.y = init_y # k-1和k-2时刻的输出量
        self.save_u = self.u.copy() # 保存历史u
        self.save_y = self.y.copy() # 保存历史y
        self.sigma = sigma  # 噪声参数
        self.na = len(A)-1  # A的待求参数量，仅用于无模型控制
        self.nb = len(B)    # B的待求参数量，仅用于无模型控制

    def controlWithU(self, u=None):
        """
        在输入为u的情况下，真实模型产生的真实数据
        """
        Yk=(1/self.A[0])*(-self.A[1]*self.y[0]-self.A[2]*self.y[1]+self.B[0]*self.u[0]+self.B[1]*self.u[1]+ random.random()*self.sigma)   #噪声
        print('reached state',Yk)
        if u!=None:
            self.u[1]=self.u[0]
            self.u[0]=u
            self.save_u.append(u)
        self.y[1]=self.y[0]
        self.y[0]=Yk
        self.save_y.append(Yk)
    
    def getCurrentY(self):
        """
        观测当前的状态Yk
        """
        Yk=(1/self.A[0])*(-self.A[1]*self.y[0]-self.A[2]*self.y[1]+self.B[0]*self.u[0]+self.B[1]*self.u[1]+ random.random()*self.sigma)
        return Yk
        
    def setTarget(self,target=10):
        """
        读取当前目标值
        """
        self.target=target
    
    def createGPC(self, theta0=[1.5,-0.7, 1.0, 0.5], beta=1, N=10, Nu=5, lamda=0.3, alpha=0.2):
        """
        创建GPC模型
        """
        self.GPC_MODEL = GPC(theta0, beta, N, Nu, lamda, alpha, self.u, self.y, self.na, self.nb)
    
    def calOutput(self, Yk, has_model=False):
        """
        确定使用有模型控制还是无模型控制
        """
        self.GPC_MODEL.setYk(Yk)
        if has_model:
            self.GPC_MODEL.setModel(self.A, self.B)
        else:
            flag, info = self.GPC_MODEL.calModel(self.save_y,self.save_u)
            if flag == False:
                print(info)
                return self.save_u[-1]
        self.GPC_MODEL.setTargetTraj(self.target)
        u = self.GPC_MODEL.calU()
        return u
    
    def show(self, target):
        sim_time = [i*0.1 for i in range(200)]
        plt.figure()
        plt.plot(sim_time, target)
        plt.plot(sim_time, self.save_y[2:])
        plt.show()


if __name__ == "__main__":
    fb = []
    for i in range(4):
        if i%2==0:
            fb.extend([0 for j in range(50)])
        else:
            fb.extend([20 for j in range(50)])
    # print(fb)


    sim_env = env(A=[1,-1.5,0.7], B=[1,0.5], init_u=[0,1], init_y=[0,0.2], sigma=0.01)
    sim_env.createGPC(theta0=[1.5,-0.7, 1.0, 0.5], beta=1, N=10, Nu=5, lamda=0.3, alpha=0.2)
    for i in range(len(fb)):
        sim_env.setTarget(fb[i])
        Yk = sim_env.getCurrentY()
        u = sim_env.calOutput(Yk, has_model=True)
        # u = sim_env.calOutput(Yk, has_model=False)
        sim_env.controlWithU(u)
    sim_env.show(fb)
