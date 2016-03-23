from collections import defaultdict
import numpy as np

cdef random_normal_limit(float arg1, float arg2):
    cdef float val   
    val = np.random.normal(arg1,arg2)
    while val > 1 or val < 0:
        val = np.random.normal(arg1,arg2)
    return val

cdef random_gamma_limit(float arg1, float arg2):
    cdef float val   
    val = np.random.gamma(arg1,arg2)
    while val > 1 or val < 0:
        val = np.random.gamma(arg1,arg2)
    return val

cdef class Grader(object):
    cdef unicode name
    cdef list handins
    cdef float bias_mean, bias_tau, bias_val

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

cdef class Handin:
    cdef unicode title
    cdef Grader owner
    cdef dict gradeings
    cdef list graders
    cdef float true_val, precision

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
        self.gradeings[grader.name] = random_normal_limit(self.true_val+bias,np.sqrt(1.0/self.precision))
        
        
cdef class Assignment(object):
    cdef dict handins, graders
    cdef Handin handin
    cdef Grader grader

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
            
cdef class Course(object):

    
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


def gibbs_model(data, int samples, int burn_in):

    # Counts
    cdef int N_H = len(data.handins) # Number of handins
    cdef int N_G = len(data.graders) # Number of graders
    cdef int N_g = data.n_gradings   # Number of gradings
    cdef int N_eval = N_g*N_G   # Number of evaluations in total
    
    # Hyperparameters
    cdef float ga_h = 0.5
    cdef float la_h = 1.0
    cdef float al_h = 10.0
    cdef float be_h = 0.1

    cdef float ga_g = 0.0
    cdef float la_g = 1.0
    cdef float al_g = 50.0
    cdef float be_g = 0.1
    
    cdef float al_e = 10.0
    cdef float be_e = 1.0
    
    # Prior parameters
    cdef dict u_h = dict()
    cdef dict t_h = dict()
    cdef dict u_g = dict()
    cdef dict t_g = dict()
    cdef dict T = dict()
    cdef dict B = dict()

    # Draw from priors
    cdef float e = np.random.gamma(al_e,1.0/be_e)
    cdef int h, g
    for h in range(N_H):
        t_h[h] = np.random.gamma(al_h,1.0/be_h)
        u_h[h] = np.random.normal(ga_h,np.sqrt(1.0/(la_g * t_h[h])))
        T[h] = np.random.normal(u_h[h],np.sqrt(1.0/t_h[h]))
    for g in range(N_G):
        t_g[g] = np.random.gamma(al_g,1.0/be_g)
        u_g[g] = np.random.normal(ga_g,np.sqrt(1.0/(la_g * t_g[g])))
        B[g] = np.random.normal(u_g[g],np.sqrt(1.0/t_g[g]))

    # Gibbs sampling
    
    # Tracers initialising
    cdef list trace_e = list()
    trace_u_h = defaultdict(list)
    trace_t_h = defaultdict(list)
    trace_u_g = defaultdict(list)
    trace_t_g = defaultdict(list)
    trace_T = defaultdict(list)
    trace_B = defaultdict(list)
    cdef int r, n_gradings
    cdef float sum_, v, la_, be_, al_
    for r in range(burn_in + samples):
        print r
        # Sample T
        for h in range(N_H):
            handin = data.handins[str(h)]
            n_gradings = len(handin.graders)
            sum_ = 0.0
            for g_n, val in handin.gradeings.iteritems():
                sum_ = sum_ + val - B[int(g_n)]
            v = e*n_gradings+t_h[h]
            T[h] = np.random.normal((u_h[h]*t_h[h]+e*sum_)/v,np.sqrt(1/v))
            
        # Sample B
        for g in range(N_G):
            grader = data.graders[str(g)]
            n_gradings = len(grader.handins)
            sum_ = 0.0
            for h_ in grader.handins:
                sum_ = sum_ + h_.gradeings[str(g)] - T[int(h_.title)]
            v = e*n_gradings+t_g[g]
            B[g] = np.random.normal((u_g[g]*t_g[g]+e*sum_)/v,np.sqrt(1/v))
        
        # Sample e
        sum_ = 0.0
        for h in range(N_H):
            for g_n, grading in data.handins[str(h)].gradeings.iteritems():
                sum_ = sum_ + np.square(grading - (T[h]+B[int(g_n)]))
        e = np.random.gamma(al_e+0.5*N_eval,1.0/(be_e+0.5*sum_))

        # Sample u_h and t_h
        for h in range(N_H):
            la_ = (la_h+1.0)
            al_ = al_h + 0.5 * la_h + 0.5 * np.square(T[h]-u_h[h])
            be_ = be_h + 0.5 + 0.5 * 1.0
#            al_ = al_h+0.5
#            be_ = be_h+0.5*((la_h*np.square(T[h]-ga_h))/la_)
            t_h[h] = np.random.gamma(al_,1.0/be_)
            u_h[h] = np.random.normal((la_h*ga_h+T[h])/la_,np.sqrt(1.0/(la_*t_h[h])))

        # Sample u_g and t_g
        for g in range(N_G):
            la_ = (la_g+1)
            al_ = al_g + 0.5 * la_g + 0.5 * np.square(B[g]-u_g[g])
            be_ = be_g + 0.5 + 0.5 * 1
#            al_ = al_g+0.5
#            be_ = be_g+0.5*((la_g*np.square(B[g]-ga_g))/la_)
            t_g[g] = np.random.gamma(al_,1.0/be_)
            u_g[g] = np.random.normal((la_g*ga_g+B[g])/la_,np.sqrt(1.0/(t_g[g])))
            
        # Collect tracings
        if r > burn_in:
            trace_e.append(e)
            for h in range(N_H):
                trace_u_h[h].append(u_h[h])
                trace_t_h[h].append(t_h[h])
                trace_T[h].append(T[h])
            for g in range(N_G):
                trace_u_g[g].append(u_g[g])
                trace_t_g[g].append(t_g[g])
                trace_B[g].append(B[g])

    traces = {'e' : trace_e,
              'u_h' : trace_u_h,
              't_h' : trace_t_h,
              'u_g' : trace_u_g,
              't_g' : trace_t_g,
              'T' : trace_T,
              'B' : trace_B}

    return traces