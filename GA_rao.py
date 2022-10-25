
import random
from turtle import onclick
import numpy as np
import math


class GA():

    def __init__(self,
        populationsize,   # 种群数量
        ND,#变量维数
        #chromosome_length,# 染色体长度
        pc,         # 交叉概率
        pm,         # 变异概率
        func,       # 待求解函数
        upper_bound,
        lower_bound,
        max_iter=200   # 迭代次数
        
    ):
        self.populationsize=populationsize
        self.ND=ND
        #self.blcoksize=20
        #self.chromosome_length=ND*self.blcoksize
        self.pc=pc
        self.pm=pm
        self.func=func
        self.max_iter=max_iter
        self.fitness_list=[] # 储存每一次迭代的最佳适应度
        self.best_ind=[]
        self.ub=upper_bound
        self.lb=lower_bound
    def generate_origin(self):
        population=[]
        # 产生populationsize个长度为chromosome_length的向量
        for i in range(self.populationsize):
            tmp=[]
            #for j in range(self.chromosome_length):
            for j in range(self.ND):
                tmp.append(random.random()*(self.ub[j]-self.lb[j])+self.lb[j])
            population.append(tmp)
        population=np.array(population)
        return population

            
    def fitall(self,population):
        fitness=[]
        for i in population:
            #将i转换为self.func需要的形式
            fitness.append(self.func(i))
        fitness=np.array(fitness)
        return fitness


    def findbest(self,population,fitness):
        val=999999
        best_index=0
        len_p=len(population)
        for i in range(len_p):
            if fitness[i]<val:
                val=fitness[i]
                best_index=i

        return population[best_index],fitness[best_index]

    def findworst(self,population,fitness):
        val=-999999
        worst_index=0
        len_p=len(population)
        for i in range(len_p):
            if fitness[i]>val:
                val=fitness[i]
                worst_index=i

        return population[worst_index],fitness[worst_index]

    def selection(self,population,fitv):
        fitv = (fitv - fitv.min()) / (fitv.max() - fitv.min() + 1e-10) + 0.2
        # the worst one should still has a chance to be selected
        sel_prob = fitv / fitv.sum()
        sel_index = np.random.choice(range(self.populationsize), size=self.populationsize, p=sel_prob)
        population = population[sel_index, :]
        return population

    def crossover_and_mutation(self,population,fitness):
        #找到本次迭代的最大值与最小值
        
        best_ind,best_fit=self.findbest(population,fitness)
        worst_ind,worst_fit=self.findworst(population,fitness)
        
        if best_fit/(worst_fit)>1000:
            r=0
        else:
            ka=-np.exp(best_fit/(worst_fit))
            r=1-(ka)/(1+ka)
        #print(type(best_ind))


        new_population=[]
        for ind in population:
            #产生交叉
            p=[]
            p.append(ind+r*(best_ind-worst_ind))
            p.append((best_ind-worst_ind)+r*(np.abs(best_ind)-np.abs(worst_ind)))
            p.append((best_ind-worst_ind)+r*(np.abs(best_ind)-worst_ind))
            #产生变异
            p.append(p[0]+0.5*np.random.rand((len(p[0]))))
            p.append(p[1]+0.5*np.random.rand((len(p[1]))))
            p.append(p[2]+0.5*np.random.rand((len(p[2]))))
            #从6个中筛选出1个加入population
            
            # print(p[0])
            # print(p[3])
            f=[]
            for i in range(6):
                f.append(self.func(p[i]))
            
            index=f.index(min(f))
            #print(p[index])
            new_population.append(p[index])
            #pass
        
        #print(new_population)
        # 从population中筛选一次
        population=np.concatenate((population,new_population),axis=0)
        #print(len(population))
        old_new=self.fitall(population)
        
        sel_index=np.argsort(old_new)[:self.populationsize]
        
        new_population=population[np.array(sel_index),:]
        #print(len(new_population))
        return new_population



    def run(self):
        

        population=self.generate_origin()
        #print(population[0])
        for i in range(self.max_iter):
            
            #计算种群的适应度
            fitness=self.fitall(population)
            #筛选一次
            #fitness=self.sift(fitness)
        
            #寻找最佳个体
            best_individul,best_fitness=self.findbest(population,fitness)
            self.fitness_list.append(best_fitness)
            self.best_ind.append(best_individul)

            population=self.selection(population,fitness)
            population=self.crossover_and_mutation(population,fitness)
            

        
        print(min(self.fitness_list))

    
def func(x):
    
    return 4*x[0]*x[0]+2.1*math.pow(x[0],4)+1/3*math.pow(x[0],6)+x[0]*x[1]-4*x[1]*x[1]+4*math.pow(x[1],4)

from matplotlib import pyplot as plt

if __name__=="__main__":
    population_size=50
    ND=2
    pc=1
    pm=0.8
    ub=5*np.ones(ND)
    lb=-5*np.ones(ND)
    ga=GA(population_size,ND,pc,pm,func,ub,lb,max_iter=200)
    # population=ga.generate_origin()
    # fitness=ga.fitall(population)
    # ga.demo(population,fitness)
    ga.run()
    plt.plot(ga.fitness_list[:200])
    plt.show()

    