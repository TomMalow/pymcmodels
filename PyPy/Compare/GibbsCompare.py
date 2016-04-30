import time
from collections import defaultdict
import numpy as np

class Grader(object):
    def __init__(self, name):
        self.name = name
        self.handins = list()
        
    def set_bias(self, mean, precision):
        self.mean = mean
        self.precision = precision
    
    def get_bias(self):
        return np.random.normal(self.mean,np.sqrt(1.0/self.precision))
        
    def add_handin(self, handin):
        self.handins.append(handin)
        
    def grade_handins(self,n_v):
        for handin in self.handins:
            handin.add_mock_gradeing(self,n_v)


class Handin:
    def __init__(self,title,owner):
        self.title = title
        self.owner = owner
        self.gradeings = dict()
        self.graders = list()
    
    def set_score(self, mean, precision):
        self.mean = mean
        self.precision = precision
    
    def add_grader(self,grader):
        self.graders.append(grader)
        
    def add_gradeing(self,grader,value):
        self.gradeings[grader.name] = value
        
    def get_score(self):
        return np.random.normal(self.mean,np.sqrt(1.0/self.precision))

    def add_mock_gradeing(self,grader,n_v):
        self.gradeings[grader.name] = np.random.normal(self.get_score()+grader.get_bias(),np.sqrt(1.0/n_v))

        
class Assignment(object):
    
    def __init__(self, handins_input, graders_input,n_gradings):
        self.handins = dict()
        self.graders = dict()
        self.n_gradings = n_gradings
        for handin in handins_input:
            self.handins[handin.title] = handin
        for grader in graders_input:
            self.graders[grader.name] = grader
    
    def add_handin(self, handin):
        self.handing[handin.title] = handin
        
    def add_grader(self, grader):
        self.graders[grader.title] = grader
        
    def grade_mock_handins(self,n_gradings,n_v):
        self.n_gradings = n_gradings
        # Distribute handins
        for i in xrange(0,n_gradings):
            for grader in self.graders.itervalues():
                h = self.find_ungraded_handin(grader)
                h.add_grader(g)
                grader.add_handin(h)
        
        # grade handins
        for grader in self.graders.itervalues():
            grader.grade_handins(n_v)
            
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

def gibbs_model(data,samples,burn_in=0):
    
    # Counts
    N_H = len(data.handins) # Number of handins
    N_G = len(data.graders) # Number of graders
    N_g = data.n_gradings   # Number of gradings
    N_eval = N_g*N_G   # Number of evaluations in total
    
    # Hyperparameters
    ga_h = 0.5
    la_h = 1.0
    al_h = 10.0
    be_h = 0.1

    ga_g = 0.0
    la_g = 1.0
    al_g = 50.0
    be_g = 0.1
    
    al_e = 10.0
    be_e = 1.0
    t_h = 500.0
    t_g = 100.0
    
    # Prior parameters
    u_h = dict()
    t_h = dict()
    u_g = dict()
    t_g = dict()
    T = dict()
    B = dict()

    # Draw from priors
    e = np.random.gamma(al_e,1.0 / be_e)
    for h in data.handins.iterkeys():
        t_h[h] = np.random.gamma(al_h,1.0/be_h)
        u_h[h] = np.random.normal(ga_h,np.sqrt(1.0/(la_h * t_h[h])))
        T[h] = np.random.normal(u_h[h],np.sqrt(1.0/t_h[h]))
    for g in data.graders.iterkeys():
        t_g[g] = np.random.gamma(al_g,1.0/be_g)
        u_g[g] = np.random.normal(ga_g,np.sqrt(1.0/(la_g * t_g[g])))
        B[g] = np.random.normal(u_g[g],np.sqrt(1.0/t_g[g]))

    # Gibbs sampling #    
    # Tracers initialising
    acc_e = list()
    acc_u_h = defaultdict(list)
    acc_t_h = defaultdict(list)
    acc_u_g = defaultdict(list)
    acc_t_g = defaultdict(list)
    acc_T = defaultdict(list)
    acc_B = defaultdict(list)
    
    tw = time.time()
    
    for r in range(burn_in + samples):
        print "\r%i" % (r+1) + " out of %i" % (burn_in + samples),
        # Sample T
        for h, handin in data.handins.iteritems():
            n_gradings = len(handin.graders)
            sum_ = 0.0
            for g, val in handin.gradeings.iteritems():
                sum_ = sum_ + val - B[g]
            v = e*n_gradings+t_h[h]
            T[h] = np.random.normal((u_h[h]*t_h[h]+e*sum_)/v,np.sqrt(1.0/v))

        # Sample B
        for g, grader in data.graders.iteritems():
            n_gradings = len(grader.handins)
            sum_ = 0.0
            for h in grader.handins:
                sum_ = sum_ + h.gradeings[g] - T[h.title]
            v = e*n_gradings+t_g[g]
            B[g] = np.random.normal((u_g[g]*t_g[g]+e*sum_)/v,np.sqrt(1.0/v))
        
        # Sample e
        sum_ = 0.0
        n_eval = 0.0
        for h, handin in data.handins.iteritems():
            for g, grading in data.handins[h].gradeings.iteritems():
                sum_ = sum_ + np.square(grading - (T[h]+B[g]))
                n_eval = n_eval + 1.0
        e = np.random.gamma(al_e+0.5*n_eval,1.0/(be_e+0.5*sum_))
        
        # Sample u_h and t_h
        for h in data.handins.iterkeys():
            la_ = (la_h+1.0)
            al_ = al_h+0.5
            be_ = be_h+0.5*((la_h*np.square(T[h]-ga_h))/la_)
            t_h[h] = np.random.gamma(al_,1.0/be_)
            u_h[h] = np.random.normal((la_h*ga_h+T[h])/la_,np.sqrt(1.0 / (la_*t_h[h])))

        # Sample u_g and t_g
        for g in data.graders.iterkeys():
            la_ = (la_g+1.0)
            al_ = al_g+0.5
            be_ = be_g+0.5*((la_g*np.square(B[g]-ga_g))/la_)
            t_g[g] = np.random.gamma(al_,1.0/be_)
            u_g[g] = np.random.normal((la_g*ga_g+B[g])/la_,np.sqrt(1.0 / (la_*t_g[g])))
        
        # Collect tracings
        if r > burn_in:
            acc_e.append(e)
            for h in data.handins.iterkeys():
                acc_u_h[h].append(u_h[h])
                acc_t_h[h].append(t_h[h])
                acc_T[h].append(T[h])
            for g in data.graders.iterkeys():
                acc_u_g[g].append(u_g[g])
                acc_t_g[g].append(t_g[g])
                acc_B[g].append(B[g])    
                
    print
    print "Wall time: %f" % (time.time() - tw)
    
    traces = {'e' : acc_e,
              'u_h' : acc_u_h,
              't_h' : acc_t_h,
              'u_g' : acc_u_g,
              't_g' : acc_t_g,
              'T' : acc_T,
              'B' : acc_B}

    return traces

def tau_std(value):
    return np.sqrt(1.0 / value)

handins_data = list()
graders_data = list()

graders = 100
gradings = 5

for i in xrange(graders):
    g = Grader('grader_%i' % i)
    g.set_bias(np.random.normal(0,tau_std(100)),np.random.gamma(50,1.0 / 0.1))
    h_mu = np.random.normal(0.5,tau_std(25))
    while h_mu > 1 or h_mu < 0:
        h_mu = np.random.normal(0.5,tau_std(25))
    h = Handin('handin_%i' % i, g)
    h.set_score(np.random.normal(0.5,tau_std(25)),np.random.gamma(10,1.0/ 0.1))
    graders_data.append(g)
    handins_data.append(h)
    
mock_data = Assignment(handins_data,graders_data,graders*gradings)
mock_data.grade_mock_handins(gradings,np.random.gamma(50,1.0 / 0.1))

mock_result = gibbs_model(mock_data,1000)