import random
import csv
import os.path


from XCSEnvironment import *
from XCSClassifier import *
from XCSClassifierSet import *
from XCSMatchSet import *
from XCSActionSet import *
from XCSConfig import *

class XCS:
    def __init__(self):
        self.env = XCSEnvironment()
        self.perf = []
        self.data=[[]]
        self.MUX=conf.MUX
        self.File=conf.File
        self.position=0
        self.max_iterations=conf.max_iterations

    def init(self):
        self.env = XCSEnvironment()
        self.perf = []
        self.position=0

    def run_experiments(self):
        if not self.MUX:
            f = open(self.File, 'r')
    
            for row in csv.reader(f):
                not_null= False
                for ele in row:
                    if ele != '':
                        not_null= True
                        break;
                if not_null:
                    self.data.append(row)
            for i in range(2):
                del(self.data[0])
            for i in range(len(self.data)):
                for j in range(4):
                    del self.data[i][0]
            for i in range(len(self.data)):
                for j in range(len(self.data[i])):
                    if self.data[i][j] == '':
                        self.data[i][j]='#'
            self.max_iterations=len(self.data)*self.max_iterations

        for exp in range(conf.max_experiments):
            random.seed(exp)
            self.actual_time = 0.0
            self.pop = XCSClassifierSet(self.env,self.actual_time)
            self.init()
            print("now"+str(exp))
            for iteration in range(self.max_iterations):
                self.run_explor(iteration,self.data)
                self.run_exploit(iteration, self.data)
            self.file_writer(exp)
            self.performance_writer(exp)
   
    def run_explor(self, iteration, data):
        if self.MUX:
            temp=iteration
            while(temp>=len(data)):
                temp=temp-len(data)
                self.position=temp
        self.env.set_state(self.MUX, data, self.position)
        self.match_set=XCSMatchSet(self.pop, self.env, self.actual_time)
        self.generate_prediction_array()
        self.select_action()
        self.action_set = XCSActionSet(self.match_set, self.action, self.env, self.actual_time)
        self.action_set.do_action()
        self.action_set.update_action_set()
        self.action_set.do_action_set_subsumption(self.pop)
        self.run_GA()
        if len(self.pop.cls)>conf.N:
            self.pop.delete_from_population()
        self.actual_time += 1.0
    
    def run_exploit(self,iteration, data):
        if iteration%100 == 0:
            if self.MUX:
                temp=random.randrange(len(data))
                self.position=temp
            p = 0
            for i in range(100):
                self.env.set_state(self.MUX, data, self.position)
                self.match_set = XCSMatchSet(self.pop,self.env,self.actual_time)
                self.generate_prediction_array()
                self.action = self.best_action()
                if self.env.is_true(self.action):
                    p += 1
            self.perf.append(p)
 
    def file_writer(self, num):
        dir_name = "Population";
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = open(dir_name+"/Population"+str(num)+".csv",'w')
        write_csv = csv.writer(file_name,lineterminator='\n')
        write_csv.writerow(["condition","action","fitness","prediction","error","numerosity","experience","time_stamp","action_set_size"])
        for cl in self.pop.cls:
            cond = ""
            for c in cl.condition:
                cond += str(c)
            write_csv.writerow([cond,cl.action,cl.fitness,cl.prediction,cl.error,cl.numerosity,cl.experience,cl.time_stamp,cl.action_set_size])
    
    def performance_writer(self,num):
        dir_name = "Performance";
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = open(dir_name+"/Performance"+str(num)+".csv",'w')
        spamWriter = csv.writer(file_name,lineterminator='\n')
        spamWriter.writerow(self.perf)
                       
    def generate_prediction_array(self):
        self.p_array=[0,0]
        self.f_array=[0,0]
        for cl in self.match_set.cls:
            self.p_array[cl.action] += cl.prediction*cl.fitness
            self.f_array[cl.action] += cl.fitness
        for i in range(2):
            if self.f_array[i] != 0:
                self.p_array[i]/=self.f_array[i]
    
    def select_action(self):
        if random.random()>conf.p_explr:
            self.action = self.best_action()
        else:
            self.action = random.randrange(2)
            
    def best_action(self):
        big = self.p_array[0]
        best = 0
        for i in range(2):
            if big < self.p_array[i]:
                big = self.p_array[i]
                best = i       
        return best
    
    def run_GA(self):
        if self.actual_time - self.action_set.ts_num_sum()/self.action_set.numerosity_sum()>conf.theta_ga:
            for cl in self.action_set.cls:
                cl.time_stamp = self.actual_time
            parent1 = self.select_offspring()
            parent2 = self.select_offspring()
            child1 = parent1.deep_copy(self.actual_time)
            child2 = parent2.deep_copy(self.actual_time)
            child1.numerosity = 1
            child2.numerosity = 1
            child1.experience = 0
            child2.experience = 0
            if random.random() <conf.chi:
                self.apply_crossover(child1, child2)
                child1.prediction = (parent1.prediction+parent2.prediction)/2.0
                child1.error = 0.25*(parent1.error+parent2.error)/2.0
                child1.fitness = 0.1*(parent1.fitness+parent2.fitness)/2.0
                child2.prediction = child1.prediction
                child2.error = child1.error
                child2.fitness = child1.fitness
            self.apply_mutation(child1)
            self.apply_mutation(child2)
            if conf.doGASubsumption:
                if parent1.does_subsume(child1):
                    parent1.numerosity += 1
                elif parent2.does_subsume(child1):
                    parent2.numerosity += 1
                else:
                    self.pop.instert_in_population(child1)
                if parent1.does_subsume(child2):
                    parent1.numerosity += 1
                elif parent2.does_subsume(child2):
                    parent2.numerosity += 1
                else:
                    self.pop.instert_in_population(child2)
            else:
                self.pop.instert_in_population(child1)
                self.pop.instert_in_population(child2)
            while self.pop.numerosity_sum() >conf.N:
                self.pop.delete_from_population()  
    
    def select_offspring(self):
        fit_sum = self.action_set.fitness_sum()
        choice_point = fit_sum * random.random()
        fit_sum = 0
        for cl in self.action_set.cls:
            fit_sum += cl.fitness
            if fit_sum >choice_point:
                return cl
        return None  

    def apply_crossover(self, cl1, cl2):
        if self.MUX:
            length = len(cl1.condition)
            sep1 = int(random.random()*(length))
            sep2 = int(random.random()*(length))
            if sep1 >sep2:
                sep1, sep2 = sep2, sep1
            elif sep1 == sep2:
                sep2 = sep2 + 1
            cond1 = cl1.condition
            cond2 = cl2.condition
            for i in range(sep1, sep2):
                if cond1[i]!=cond2[i]:
                    cond1[i],cond2[i] = cond2[i], cond1[i]
            cl1.condition =cond1
            cl2.condition =cond2
        else:
            for i in range(len(cl1.condition)):
                
        
    def apply_mutation(self,cl):
        i = 0
        for i in range(len(cl.condition)):
            if random.random() < conf.myu:
                if cl.condition[i] == '#':
                    if self.env.state[i] == '#':
                        cl.condition[i]=random.randrange(1,5)
                    else:
                        cl.condition[i] = self.env.state[i]
                else:
                    cl.condition[i] = '#'
        if random.random() <conf.myu:
            cl.action = random.randrange(2)

if __name__ == '__main__':
    xcs = XCS()
    xcs.run_experiments()
