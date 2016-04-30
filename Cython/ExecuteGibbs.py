from collections import defaultdict
import numpy as np
import time
import GibbsCython

def random_limit(distribution, arg1, arg2):
    
    val = distribution(arg1,arg2)
    while val > 1 or val < 0:
        val = distribution(arg1,arg2)
    return val

class Grader(object):
    def __init__(self, name,bias_mean,bias_tau):
        self.name = name
        self.handins = list()
        self.bias_mean = bias_mean
        self.bias_tau = bias_tau
        
    def add_handin(self, handin):
        self.handins.append(handin)
                
    def grade_handins(self):
        for handin in self.handins:
            bias_val = np.random.normal(self.bias_mean,np.sqrt(1.0/self.bias_tau))
            handin.add_gradeing(self,bias_val)

class Handin:
    def __init__(self,title,owner,true_value,precision):
        self.title = title
        self.owner = owner
        self.gradeings = dict()
        self.graders = list()
        self.true_val = true_value
        self.precision = precision
    
    def add_grader(self,grader):
        self.graders.append(grader)
    
    def add_gradeing(self,grader,bias):
        self.gradeings[grader.name] = random_limit(np.random.normal,self.true_val+bias,np.sqrt(1.0/self.precision))
        
        
class Assignment(object):
    
    def __init__(self, handins_input, graders_input):
        self.handins = dict()
        self.graders = dict()
        for handin in handins_input:
            self.handins[handin.title] = handin
        for grader in graders_input:
            self.graders[grader.name] = grader
    
    def add_handin(self, handin):
        self.handing[handin.title] = handin
        
    def add_grader(self, grader):
        self.graders[grader.title] = grader
    
    def find_ungraded_handin(self, grader):
        
        # sort the handins by the one with the least
        sorted_l = sorted(self.handins.values(),key=lambda x: len(x.graders))
        #i = int(random.uniform(0,len(sorted_l)))
        i = 0
        handin = sorted_l[i]
        while handin in grader.handins or (handin.owner.name == grader.name):
        #while(handin.owner.name == grader.name):
            i += 1
            #i = int(random.uniform(0,len(sorted_l)))
            handin = sorted_l[i]
        return handin
            
    def grade_handins(self,n_gradings):
        self.n_gradings = n_gradings
        # Distribute handins
        for i in xrange(0,n_gradings):
            for grader in self.graders.itervalues():
                h = self.find_ungraded_handin(grader)
                h.add_grader(grader)
                grader.add_handin(h)
                
        # grade handins
        for grader in self.graders.itervalues():
            grader.grade_handins()
            
class Course(object):
    
    def __init__(self):
        self.assignments = list()
        self.handins = dict()
        self.graders = dict()
        self.n_gradings = 0
    
    def add_assignment(self,assignment):
        self.assignments.append(assignment)
        for a in self.assignments:
            self.handins.update(a.handins)
            self.graders.update(a.graders)
        self.n_gradings = self.n_gradings + a.n_gradings

course = Course()

handins_data = list()
graders_data = list()

for i in xrange(0,100):
    mu = np.random.normal(0.0,np.sqrt(1.0/100.0))
    tau = np.random.gamma(50.0,1.0/0.1)
    g = Grader('%i' % i,mu,tau)
    graders_data.append(g)
for i in xrange(0,20):
    t_mu = random_limit(np.random.normal,0.5,np.sqrt(1.0/25.0))
    t_tau = np.random.gamma(10,np.sqrt(1.0/0.1))
    h = Handin('%i' % i, graders_data[i], t_mu, t_tau)
    handins_data.append(h)
    
assignment_data = Assignment(handins_data,graders_data)
assignment_data.grade_handins(5)
course.add_assignment(assignment_data)

#handins_data = list()
#
#for i in xrange(100,200):
#    t_mu = random_limit(np.random.normal,0.5,np.sqrt(1.0/25.0))
#    t_tau = np.random.gamma(10,np.sqrt(1.0/0.1))
#    h = Handin('%i' % i, graders_data[i-100], t_mu, t_tau)
#    handins_data.append(h)
    
#assignment_data_2 = Assignment(handins_data,graders_data)
#assignment_data_2.grade_handins(5)
#course.add_assignment(assignment_data_2)

#handins_data = list()

#for i in xrange(200,300):
#    t_mu = random_limit(np.random.normal,0.5,np.sqrt(1.0/25.0))
#    t_tau = np.random.gamma(10,np.sqrt(1.0/0.1))
#    h = Handin('%i' % i, graders_data[i-300], t_mu, t_tau)
#    handins_data.append(h)
    
#assignment_data_3 = Assignment(handins_data,graders_data)
#assignment_data_3.grade_handins(5)
#course.add_assignment(assignment_data_3)

#for i in xrange(300,400):
#    t_mu = random_limit(np.random.normal,0.5,np.sqrt(1.0/25.0))
#    t_tau = np.random.gamma(10,np.sqrt(1.0/0.1))
#    h = Handin('%i' % i, graders_data[i-400], t_mu, t_tau)
#    handins_data.append(h)
    
#assignment_data_4 = Assignment(handins_data,graders_data)
#assignment_data_4.grade_handins(5)
#course.add_assignment(assignment_data_4)

tw = time.time()

traces = GibbsCython.gibbs_model(course,1000,0)

print "Wall time: %f" % (time.time() - tw)