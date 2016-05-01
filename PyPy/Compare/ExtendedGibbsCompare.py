import time
from collections import defaultdict
import numpy as np

class Grader_(object):
    def __init__(self, id):
        self.id = id
        self.handins = list()

    def set_bias(self, mean, precision):
        self.mean = mean
        self.precision = precision
    
    def get_bias(self):
        return np.random.normal(self.mean,np.sqrt(1.0/self.precision))
        
    def add_handin(self, handin):
        self.handins.append(handin)

    def grade_handins(self,questions,n_v):
        for handin in self.handins:
            for question in questions.itervalues():
                handin.add_mock_gradeing(question,self,n_v)

class Question_(object):
    def __init__(self,id):
        self.id = id

class Answer_(object):
    def __init__(self,question,grader,value):
        self.question = question
        self.grader = grader
        self.value = value
        
    def set_mock_value(self,mean,precision):
        self.mean = mean
        self.precision = precision

    def set_mock_score(self,score):
        self.valeu = score
    
    def get_score(self):
        return np.random.normal(self.mean,np.sqrt(1.0/self.precision))

class Gradeings_(object):
    def __init__(self,grader):
        self.grader = grader
        self.answers = dict()

    def add_answer(self,question,answer):
        self.answer[question] = value

    def set_answers_scores(self):
        for answer in self.answers.itervalues():
            h_mu = np.random.normal(0.5,tau_std(25))
            answer.set_mock_value(h_mu,np.random.gamma(10,1.0/ 0.1))

    def add_mock_gradeing(self,question,grader,n_v):
        a = Answer_(question,grader,0)
        h_mu = np.random.normal(0.5,tau_std(25))
        while h_mu > 1 or h_mu < 0:
            h_mu = np.random.normal(0.5,tau_std(25))
        a.set_mock_value(h_mu,np.random.gamma(10,1.0/ 0.1))

        mock_score = np.random.normal(a.get_score()+grader.get_bias(),np.sqrt(1.0/n_v))
        a.set_mock_score(mock_score)
        self.answers[question.id] = a

class Handin_(object):
    def __init__(self,id,owner):
        self.id = id
        self.owner = owner
        self.gradeings = dict()
        self.graders = list()
        
    def set_answers(self,question,grader,answers):
        for answer in answers:
            self.gradeing[grader.id].add_answer(question.id,value)
        
    def set_answers_scores(self):
        for gradeing in self.gradeings.itervalues():
            gradeing.set_answers_scores()

    def add_grader(self,grader):
        self.graders.append(grader)
        
    def add_answer(self,grader,question,value):
        if str(question.id) not in self.answers:
            self.answers[str(question.id)] = dict()
        self.answers[str(question.id)][str(grader.id)] = value
        
    def get_graders_answers(self):
        graders_answers = defaultdict(list)
        for q, answer in self.answers.iteritems():
            for g, value in answer.iteritems():
                graders_answers[g].append((q,value))
            
        return graders_answers

    def add_mock_gradeing(self,question,grader,n_v):
        if grader.id not in self.gradeings:
            self.gradeings[grader.id] = Gradeings_(grader)
        self.gradeings[grader.id].add_mock_gradeing(question,grader,n_v)
        
    def get_grader_answers(self,grader):
        grader_answers = list()
        for key, values in self.answers.iteritems():
            grader_answers.append((key,values[grader]))
            
        return grader_answers

class Assignment_(object):
    
    def __init__(self, handins_input, graders_input, questions_input, n_gradings):
        self.graders = dict()
        self.handins = dict()
        self.questions = dict()
        self.n_gradings = n_gradings
        for handin in handins_input:
            self.handins[handin.id] = handin
        for grader in graders_input:
            self.graders[grader.id] = grader
        for question in questions_input:
            self.questions[question.id] = question

    def add_handin(self, handin):
        self.handing[handin.id] = handin
        
    def add_grader(self, grader):
        self.graders[grader.id] = grader
    
    def set_questions(self, questions):
        self.questions = list(questions)
        
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
            grader.grade_handins(self.questions,n_v)
            
    def find_ungraded_handin(self, grader):
        
        # sort the handins by the one with the least
        sorted_l = sorted(self.handins.values(),key=lambda x: len(x.graders))
        #i = int(random.uniform(0,len(sorted_l)))
        i = 0
        handin = sorted_l[i]
        while handin in grader.handins or (handin.owner.id == grader.id):
        #while(handin.owner.name == grader.name):
            i += 1
            #i = int(random.uniform(0,len(sorted_l)))
            handin = sorted_l[i]
        return handin

def gibbs_model(data, samples, burn_in=0):
    
    # Counts
    N_H = len(data.handins) # Number of handins
    N_G = len(data.graders) # Number of graders
    N_Q = len(data.questions) # Number of graders
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
    
    al_n = 10.0
    be_n = 1.0
    
    # Prior parameters
    u_h = dict()
    t_h = dict()
    T = defaultdict(dict)
    B = defaultdict(dict)
    u_g = dict()
    t_g = dict()

    # Draw from priors
    n_v = np.random.gamma(al_n,1.0/be_n)
    for h, handin in data.handins.iteritems():
        t_h[h] = np.random.gamma(al_h,1.0/be_h)
        u_h[h] = np.random.normal(ga_h,np.sqrt(1.0/(la_h * t_h[h])))
        for q in data.questions.iterkeys():        
            T[h][q] = np.random.normal(u_h[h],np.sqrt(1.0/t_h[h]))
    for g, grader in data.graders.iteritems():
        t_g[g] = np.random.gamma(al_g,1.0/be_g)
        u_g[g] = np.random.normal(ga_g,np.sqrt(1.0/(la_g * t_g[g])))
        for q in data.questions.iterkeys():
            B[g][q] = np.random.normal(u_g[g],np.sqrt(1.0/t_g[g]))

    # Gibbs sampling #
    
    # Tracers initialising
    acc_n_v = list()
    acc_u_h = defaultdict(list)
    acc_t_h = defaultdict(list)
    acc_u_g = defaultdict(list)
    acc_t_g = defaultdict(list)
    acc_T = defaultdict(lambda: defaultdict(list))
    acc_B = defaultdict(lambda: defaultdict(list))

    tw = time.time()
    for r in range(burn_in + samples):
        print "\r%i" % (r+1) + " out of %i" % (burn_in + samples),
        # Sample T
        for h, handin in data.handins.iteritems():
            for g, grade in handin.gradeings.iteritems():
                n_gradings = len(grade.answers)
                sum_ = 0.0
                for q, val in grade.answers.iteritems():
                    sum_ = sum_ + val.value - B[g][q]
                v = n_v*n_gradings+t_h[h]
                T[h][q] = np.random.normal((u_h[h]*t_h[h]+n_v*sum_)/v,np.sqrt(1.0/v))
            
        # Sample B
        for g, grader in data.graders.iteritems():
            for q in data.questions.iterkeys():
                n_gradings = len(grader.handins)
                sum_ = 0.0
                for h in grader.handins:
                    sum_ = sum_ + h.gradeings[g].answers[q].value - T[h.id][q]
                v = n_v * n_gradings + t_g[g]
                B[g][q] = np.random.normal((u_g[g]*t_g[g]+n_v*sum_)/v,np.sqrt(1.0/v))
        
        # Sample e
        sum_ = 0.0
        n_eval = 0
        for h, handin in data.handins.iteritems():
            for g, grade in handin.gradeings.iteritems():
                for q, answer in grade.answers.iteritems():
                    n_eval = n_eval + 1
                    sum_ = sum_ + np.square(answer.value - (T[h][q]+B[g][q]))
        n_v = np.random.gamma(al_n+0.5*n_eval,1.0 / (be_n+0.5*sum_))

        # Sample u_q and t_q
        for h in data.handins.iterkeys():
            la_ = (la_h+N_Q)
            sum_q = 0.0
            for q in data.questions.iterkeys():
                sum_q = sum_q = T[h][q]
            mean_q = sum_q / N_Q
            sum_minus = 0.0
            for q in data.questions.iterkeys():
                sum_minus = sum_minus + np.square(T[h][q]-mean_q)
            al_ = al_h+0.5*N_Q
            be_ = be_h+0.5*(N_Q*sum_minus+(N_Q*la_h*np.square(mean_q-ga_h))/la_)
            t_h[h] = np.random.gamma(al_,1.0 / be_)
            u_h[h] = np.random.normal((la_h*ga_h+sum_q)/la_,np.sqrt(1.0/(la_*t_h[h])))

        # Sample u_g and t_g
        for g in data.graders.iterkeys():
            la_ = (la_g+N_Q)
            sum_q = 0.0
            for q in data.questions.iterkeys():
                sum_q = sum_q + B[g][q]
            mean_q = sum_q / N_Q
            sum_minus = 0.0
            for q in data.questions.iterkeys():
                sum_minus = sum_minus + np.square(B[g][q]-mean_q)
            al_ = al_g+0.5*N_Q
            be_ = be_g+0.5*(N_Q*sum_minus+(N_Q*la_g*np.square(mean_q-ga_g))/la_)
            t_g[g] = np.random.gamma(al_,1.0 / be_)
            u_g[g] = np.random.normal((la_g*ga_g+sum_q)/la_,np.sqrt(1.0/(la_*t_g[g])))
                        
        # Collect tracings
        if r > burn_in:
            acc_n_v.append(n_v)
            for h in data.handins.iterkeys():
                acc_u_h[h].append(u_h[h])
                acc_t_h[h].append(t_h[h])
                for q in data.questions.iterkeys():
                    acc_T[h][q].append(T[h][q])
            for g in data.graders.iterkeys():
                acc_u_g[g].append(u_g[g])
                acc_t_g[g].append(t_g[g])
                for q in data.questions.iterkeys():
                    acc_B[g][q].append(B[g][q])
                    
    print
    print "Wall time: %f" % (time.time() - tw)
    
    traces = {'n_v' : acc_n_v,
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
questions_data = list()
answers_data = list()

graders = 60
questions = 20
gradings = 5

for i in xrange(questions):
    q = Question_('question_%i' % i)
    questions_data.append(q)

for i in xrange(graders):
    g = Grader_('grader_%i' % i)
    g.set_bias(np.random.normal(0,tau_std(100)),np.random.gamma(50,1.0 / 0.1))
    h = Handin_('handin_%i' % i, g)
    h.set_answers_scores()
    graders_data.append(g)
    handins_data.append(h)
    
mock_data = Assignment_(handins_data,graders_data,questions_data,graders*gradings)
mock_data.grade_mock_handins(gradings,np.random.gamma(50,1.0 / 0.1))

mock_result = gibbs_model(mock_data,0,1000)