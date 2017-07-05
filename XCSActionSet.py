import random

from XCSConfig import *
from XCSEnvironment import *
from XCSClassifier import *
from XCSClassifierSet import *
from XCSMatchSet import *

class XCSActionSet(XCSClassifierSet):
    
    def __init__(self, match_set, action, env, actual_time):
        XCSClassifierSet.__init__(self, env, actual_time)
        self.action = action
        for cl in match_set.cls:
            if cl.action == self.action:
                self.cls.append(cl)
        
    def do_action(self):
        if self.env.is_true(self.action):
            self.reward = 1000
        else:
            self.reward = 0
    
    def update_action_set(self):
        num_sum = self.numerosity_sum()
        for cl in self.cls:
            cl.update_parameters(self.reward, num_sum)
        acc_sum = self.accuracy_sum()
        for cl in self.cls:
            cl.update_fitness(acc_sum)
    
    def do_action_set_subsumption(self,pop):
        if conf.doActionSetSubsumtion:
            subsumer = None
            for cl in self.cls:
                if cl.cloud_subsume():
                    if subsumer == None or cl.is_more_general(subsumer):
                        subsumer = cl
            if subsumer != None:
                i=0
                while i<len(self.cls):
                    if subsumer.is_more_general(self.cls[i]):
                        subsumer.numerosity += self.cls[i].numerosity
                        pop.remove_classifier_by_instance(self.cls[i])
                        self.remove_classifier(i)
                        i -= 1
                    i += 1        
