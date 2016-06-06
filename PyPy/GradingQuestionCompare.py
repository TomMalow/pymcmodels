# Add the application folder to the path
import sys
import time
import math
from collections import defaultdict
import numpy as np
#import pymc as pm

def tau_std(value):
    return np.sqrt(1.0 / value)

class Grader_(object):
    def __init__(self, id):
        self.id = id
        self.handins = list()
    def set_bias(self, mean, precision):
        self.mean = mean
        self.precision = precision
    
    def get_bias(self):
        return np.random.normal(self.mean,tau_std(self.precision))
        
    def add_handin(self, handin):
        self.handins.append(handin)

    def grade_handins(self,questions,n_v):
        for handin in self.handins:
            for question in questions.itervalues():
                handin.add_mock_gradeing(question,self,n_v)
                
class Question_(object):
    def __init__(self,id):
        self.id = id
                
class Answers_(object):
    def __init__(self,handin,question):
        self.handin = handin
        self.question = question
        self.answers = dict()
    
    def set_mock_value(self,mean,precision):
        self.mean = mean
        self.precision = precision
        
    def set_mock_score(self,grader,score):
        self.answer[str(question)] = score

    def get_score(self):
        return self.mean

class Handin_(object):
    def __init__(self,id,owner):
        self.id = id
        self.owner = owner
        self.gradings = dict()
        self.graders = list()
        self.catched_score = dict()
        self.precision = np.random.gamma(10,1.0/ 0.1)
        while 1:
            h_mu = np.random.normal(0.5,tau_std(np.random.gamma(2,1.0/ 0.1)))
            if h_mu < 1.0 and h_mu > 0.0:
                break
        self.mean = h_mu
            
    def set_answers_scores(self):
        for gradings in self.gradings.itervalues():
            while 1:
                h_mu = np.random.normal(self.mean,tau_std(20.0))
                if h_mu < 1.0 and h_mu > 0.0:
                    break
            answer.set_mock_score(h_mu,np.random.gamma(1,1.0/ 0.1))
            
    def add_grader(self,grader):
        self.graders.append(grader)
        
    def add_answer(self,grader,question,value):
        if str(question.id) not in self.gradings:
            self.gradings[str(question.id)] = Answers_(self,grader)
        self.gradings[str(question.id)].answers[str(grader.id)] = value
        
    def get_graders_answers(self):
        graders_answers = defaultdict(list)
        for q, answer in self.gradings.iteritems():
            for g, value in answer.answers.iteritems():
                graders_answers[str(g)].append((str(q),value))
            
        return graders_answers
        
    def get_grader_answers(self, grader):
        grader_answers = list()
        for key, answer in self.gradings.iteritems():
            grader_answers.append((key, answer.answers[grader]))
        return grader_answers
    
    def get_handin_score(self, g):
        if g not in self.catched_score:
            grader_g = list()
            for answers in self.gradings.itervalues():
                if g not in answers.answers:
                    return None
                grader_g.append(answers.answers[g])
            self.catched_score[g] = np.mean(grader_g)
        
        return self.catched_score[g]
    
    def add_mock_gradeing(self,question,grader,n_v):
        if question.id not in self.gradings:
            a = Answers_(self,question)
            while 1:
                a_mu = np.random.normal(self.mean,tau_std(self.precision))
                if a_mu < 1.0 and a_mu > 0.0:
                    break
            a.set_mock_value(a_mu,np.random.gamma(100,1.0/ 0.1))
            self.gradings[str(question.id)] = a
        a = self.gradings[str(question.id)]
        while 1:
            mock_value = np.random.normal(a.get_score()+grader.get_bias(),tau_std(n_v))
#            if mock_value < 1.0 and mock_value > 0.0:
            break
        a.answers[str(grader.id)] = mock_value

class Assignment_(object):
    
    def __init__(self, handins_input, graders_input, questions_input, n_gradings):
        self.graders = dict()
        self.handins = dict()
        self.questions = dict()
        self.n_gradings = n_gradings
        for handin in handins_input:
            self.handins[str(handin.id)] = handin
        for grader in graders_input:
            self.graders[str(grader.id)] = grader
        for question in questions_input:
            self.questions[str(question.id)] = question

    def add_handin(self, handin):
        self.handing[str(handin.id)] = handin
        
    def add_grader(self, grader):
        self.graders[str(grader.id)] = grader
    
    def set_questions(self, questions):
        self.questions = list(questions)
        
    def grade_mock_handins(self, n_gradings, n_v):
        self.n_gradings = n_gradings
        # Distribute handins
        for i in xrange(n_gradings):
            for grader in self.graders.itervalues():
            #for handin in self.handins.itervalues():
                #grader = self.find_grader_for_grading(handin)
                handin = self.find_ungraded_handin(grader)
                handin.add_grader(grader)
                grader.add_handin(handin)
        
        # grade handins
        for grader in self.graders.itervalues():
            grader.grade_handins(self.questions, n_v)
    
    def find_grader_for_grading(self, handin):
        sorted_l = sorted(self.graders.values(), key=lambda x: len(x.handins))
        i = 0
        grader = sorted_l[i]
        while grader in handin.graders or handin.owner.id == grader.id:
            grader = sorted_l[i]
            i += 1
        return grader

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

def user_name(user_id):
    user = data_model.User.objects.get(id=user_id)
    return user.name

def user_id(user_name):
    for user in data_model.User.objects(name=user_name):
        return user.id

def question_text(q_id):
    user = data_model.Question.objects.get(id=ObjectId(q_id))
    return user.text

def question_max_value(question):
    if question.question_type == "boolean":
        return 1
    elif question.question_type == "numerical":
        if question.numericalAnswers:
            return max(map(int,question.numericalAnswers.keys()))
        else:
            return 1

def answer_value(answer):
    if answer.numerical_answer != None:
        return answer.numerical_answer / float(question_max_value(answer.question))
    if answer.boolean_answer != None:
        return answer.boolean_answer / float(question_max_value(answer.question))

def answeres_handin(report_grade):
    '''Returns a list of tuples containing the answer and the value of the question'''
    answers = data_model.Answer.objects(report_grade=report_grade)
    result = list()
    for answer in answers:
        if answer.numerical_answer != None or answer.boolean_answer != None:
            result.append((answer,answer_value(answer)))
    return result

# def PyMC_peergrade_model(data,samples,burn_in=0):
    
#     # Initialize Containers for posterior
#     mu_h = dict()
#     tau_h = dict()
#     mu_g = dict()
#     tau_g = dict()
#     O = list()

#     for h_id, handin in data.handins.iteritems():
        
#         mu_h[h_id] = pm.Normal('mu_h_%s' % str(h_id), 0.5, 25)
#         tau_h[h_id] = pm.Gamma('tau_h_%s' % str(h_id), 10, 0.1)
        
#         for g in handin.graders:
#             val = handin.get_handin_score(g.id)
            
#             if g.id not in mu_g:
#                 mu_g[g.id] = pm.Normal('mu_g_%s' % str(g.id), 0, 100)
#                 tau_g[g.id] = pm.Gamma('tau_g_%s' % str(g.id), 50, 0.1)
                
#             O.append(pm.Normal('O_%s_%s' % (h_id, g.id),
#                                mu=mu_h[h_id] + mu_g[g.id],
#                                tau=tau_h[h_id] + tau_g[g.id],
#                                observed=True, value=val))
               
#     collection = [pm.Container(mu_g), pm.Container(tau_g),
#                   pm.Container(mu_h), pm.Container(tau_h),
#                   pm.Container(O)]
    
#     model = pm.Model(collection)
#     mcmc = pm.MCMC(model)
#     tw = time.time()
#     mcmc.sample(samples, burn_in, progress_bar=False)
#     print "Wall time: %f" % (time.time() - tw)
#     return mcmc

# Log Probabiliy density functions used in Metropolis Hasting
def norm_log_pdf(x, mu, tau):
    '''Normal Log Probability density function'''
    return -0.5 * tau * (x - mu) ** 2 + np.log(tau) \
        - np.log(np.sqrt(2.0 * math.pi))

def gamma_log_pdf(x, al, be):
    '''Gamma log Probability density function'''
    return al * np.log(be) - np.log(math.gamma(al)) + (al - 1.0) \
        * np.log(x) - be * x

def norm_gamma_log_pdf(mu, tau, ga, la, al, be):
    '''Normal-Gamma Log Probability density function'''
    return al * np.log(be) + np.log(np.sqrt(la)) \
        - np.log(math.gamma(al)) - np.log(np.sqrt(2.0 * math.pi)) \
        + (al - 1) * np.log(tau) - np.log(be * tau) - 0.5 \
        * tau * la * (mu - ga) ** 2

def MH_model(data, samples=1000, burn_in=0):
    '''Performs Metropolis Hasting sampling on an assignment
    object. Returns the found latent score for each hand-ins
    and the bias of each grader

    Keyword arguments:
    samples -- the number of samples to run
    burn-in -- added perirod to burn-in the samples.
    '''
    
    # Hyperparameters
    ga_h = 0.5
    la_h = 1.0
    al_h = 10.0
    be_h = 0.1

    ga_g = 0.0
    la_g = 1.0
    al_g = 50.0
    be_g = 0.1
    
    tau_h = 500.0
    tau_g = 100.0
    
    # Prior parameters
    mu_h = dict()
    tau_h = dict()
    mu_g = dict()
    tau_g = dict()
    
    log_h = dict()
    log_g = dict()

    # Functions for finding probabilities
    def prop_mu_tau_h(handin, _mu_h, _tau_h):
        '''Finds the priobability of mu_h and tau_h given the priors'''
        sum_ = 0.0
        for g in handin.graders:
            val = handin.get_handin_score(g.id)
            if val is not None:
                sum_ = sum_ + norm_log_pdf(val, mu_g[g.id] + _mu_h,
                                           tau_g[g.id] + _tau_h)
        return sum_ + norm_gamma_log_pdf(_mu_h, _tau_h, ga_h, la_h,
                                         al_h, be_h)
    
    def prop_mu_tau_g(grader, _mu_g, _tau_g):
        '''Finds the priobability of mu_g and tau_g given the priors'''
        sum_ = 0.0
        for h in grader.handins:
            val = h.get_handin_score(grader.id)
            if val is not None:
                sum_ = sum_ + norm_log_pdf(val, _mu_g + mu_h[h.id],
                                           _tau_g + tau_h[h.id])
        return sum_ + norm_gamma_log_pdf(_mu_g, _tau_g, ga_g, la_g,
                                         al_g, be_g)
    
    # Draw from priors
    for h in data.handins.iterkeys():
        tau_h[h] = np.random.gamma(al_h, 1 / be_h)
        mu_h[h] = np.random.normal(ga_h, np.sqrt(1 / (la_g * tau_h[h])))
    for g in data.graders.iterkeys():
        tau_g[g] = np.random.gamma(al_g, 1 / be_g)
        mu_g[g] = np.random.normal(ga_g, np.sqrt(1 / (la_g * tau_g[g])))

    # Pre-calculate the likelihood
    for h, handin in data.handins.iteritems():
        log_h[h] = prop_mu_tau_h(handin, mu_h[h], tau_h[h])
    for g, grader in data.graders.iteritems():
        log_g[g] = prop_mu_tau_g(grader, mu_g[g], tau_g[g])
            
    # Tracers initialising
    trace_mu_h = defaultdict(list)
    trace_tau_h = defaultdict(list)
    trace_mu_g = defaultdict(list)
    trace_tau_g = defaultdict(list)
    
    tw = time.time()
    for step in range(burn_in + samples):
        print "\r%i out of %i" % ((step + 1), (burn_in + samples)),
        
        # Sample mu_h and tau_h
        for h, handin in data.handins.iteritems():
            # Propose new candidates
            mu_h_can = np.random.normal(mu_h[h], 0.1)
            tau_h_can = np.random.normal(tau_h[h], 0.1)
            # Find likelihood
            p_ = prop_mu_tau_h(handin, mu_h_can, tau_h_can)
            # Calculate acception rate
            alpha = min(1, p_ - log_h[h])
            if np.log(np.random.random()) <= alpha:
                mu_h[h] = mu_h_can
                tau_h[h] = tau_h_can
                log_h[h] = p_
                    
        # Sample mu_g and tau_g
        for g, grader in data.graders.iteritems():
            # Propose new candidates
            mu_g_can = np.random.normal(mu_g[g], 0.1)
            tau_g_can = np.random.normal(tau_g[g], 0.1)
            # Find likelyhood
            p_ = prop_mu_tau_g(grader, mu_g_can, tau_g_can)
            # Calculate acception rate
            alpha = min(1, p_ - log_g[g])
            if np.log(np.random.random()) <= alpha:
                mu_g[g] = mu_g_can
                tau_g[g] = tau_g_can
                log_g[g] = p_

        # Collect tracings
        if step > burn_in:
            for h in data.handins.iterkeys():
                trace_mu_h[h].append(mu_h[h])
                trace_tau_h[h].append(tau_h[h])
            for g in data.graders.iterkeys():
                trace_mu_g[g].append(mu_g[g])
                trace_tau_g[g].append(tau_g[g])
                
    print
    print "Wall time: %f" % (time.time() - tw)
    
    traces = {'mu_h': trace_mu_h,
              'tau_h': trace_tau_h,
              'mu_g': trace_mu_g,
              'tau_g': trace_tau_g}

    return traces

def gibbs_model(data, samples=1000, burn_in=0):
    '''Performs Gibbs sampling on an assignment
    object. Returns the found latent score for each hand-ins
    and the bias of each grader

    Keyword arguments:
    samples -- the number of samples to run
    burn-in -- added perirod to burn-in the samples.
    '''
    
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
    mu_h = dict()
    tau_h = dict()
    mu_g = dict()
    tau_g = dict()
    T = dict()
    B = dict()

    # Draw from priors
    n_v = np.random.gamma(al_n, 1.0 / be_n)
    for h_id in data.handins.iterkeys():
        tau_h[h_id] = np.random.gamma(al_h, 1.0 / be_h)
        mu_h[h_id] = np.random.normal(ga_h, np.sqrt(1.0 / (la_h * tau_h[h_id])))
        T[h_id] = np.random.normal(mu_h[h_id], np.sqrt(1.0 / tau_h[h_id]))
    for g_id in data.graders.iterkeys():
        tau_g[g_id] = np.random.gamma(al_g, 1.0 / be_g)
        mu_g[g_id] = np.random.normal(ga_g, np.sqrt(1.0 / (la_g * tau_g[g_id])))
        B[g_id] = np.random.normal(mu_g[g_id], np.sqrt(1.0 / tau_g[g_id]))
    
    # Tracers initialising
    traces_n_v = list()
    traces_mu_h = defaultdict(list)
    traces_tau_h = defaultdict(list)
    traces_mu_g = defaultdict(list)
    traces_tau_g = defaultdict(list)
    traces_T = defaultdict(list)
    traces_B = defaultdict(list)

    # Execution of Gibbs sampling
    tw = time.time()
    for step in range(burn_in + samples):
        print "\r%i out of %i" % ((step + 1), (burn_in + samples)),
        
        # Sample T
        for h, handin in data.handins.iteritems():
            n_gradings = 0.0
            sum_ = 0.0
            for g in handin.graders:
                val = handin.get_handin_score(g.id)
                if val is not None:
                    n_gradings = n_gradings + 1.0
                    sum_ = sum_ + val - B[g.id]
            v = n_v * n_gradings + tau_h[h]
            T[h] = np.random.normal((mu_h[h] * tau_h[h] + n_v * sum_) / v,
                                    np.sqrt(1.0 / v))

        # Sample B
        for g, grader in data.graders.iteritems():
            n_gradings = 0.0
            sum_ = 0.0
            for handin in grader.handins:
                val = handin.get_handin_score(g)
                if val is not None:
                    n_gradings = n_gradings + 1.0
                    sum_ = sum_ + val - T[handin.id]
            v = n_v * n_gradings + tau_g[g]
            B[g] = np.random.normal((mu_g[g] * tau_g[g] + n_v * sum_) / v,
                                    np.sqrt(1.0 / v))

        # Sample n_v
        sum_ = 0.0
        n_eval = 0
        for h, handin in data.handins.iteritems():
            for grader in handin.graders:
                val = handin.get_handin_score(grader.id)
                if val is not None:
                    n_eval = n_eval + 1
                    sum_ = sum_ + (val - (T[h] + B[grader.id])**2)
        n_v = np.random.gamma(al_n + 0.5 * n_eval, 1.0 / (be_n + 0.5 * sum_))

        # Sample mu_h and tau_h
        for h in data.handins.iterkeys():
            la_ = (la_h + 1.0)
            al_ = al_h + 0.5
            be_ = be_h + 0.5 * ((la_h * (T[h] - ga_h)**2) / la_)
            tau_h[h] = np.random.gamma(al_, 1.0 / be_)
            mu_h[h] = np.random.normal((la_h * ga_h + T[h]) / la_,
                                       np.sqrt(1.0 / (la_ * tau_h[h])))

        # Sample mu_g and tau_g
        for g in data.graders.iterkeys():
            la_ = (la_g + 1.0)
            al_ = al_g + 0.5
            be_ = be_g + 0.5 * ((la_g * (B[g] - ga_g)**2) / la_)
            tau_g[g] = np.random.gamma(al_, 1.0 / be_)
            mu_g[g] = np.random.normal((la_g * ga_g + B[g]) / la_,
                                       np.sqrt(1.0 / (la_ * tau_g[g])))
                        
        # Collect tracings
        if step > burn_in:
            traces_n_v.append(n_v)
            for h in data.handins.iterkeys():
                traces_mu_h[h].append(mu_h[h])
                traces_tau_h[h].append(tau_h[h])
                traces_T[h].append(T[h])
            for g in data.graders.iterkeys():
                traces_mu_g[g].append(mu_g[g])
                traces_tau_g[g].append(tau_g[g])
                traces_B[g].append(B[g])
                    
    print
    print "Wall time: %f" % (time.time() - tw)
    
    traces = {'n_v': traces_n_v,
              'mu_h': traces_mu_h,
              'tau_h': traces_tau_h,
              'mu_g': traces_mu_g,
              'tau_g': traces_tau_g,
              'T': traces_T,
              'B': traces_B}

    return traces

def gibbs_ext_model(data, samples, burn_in=0):
    '''Performs Gibbs sampling on an assignment
    object. Returns the found latent score for each hand-ins
    and the bias of each grader

    Keyword arguments:
    samples -- the number of samples to run
    burn-in -- added perirod to burn-in the samples.
    '''
    
    # Number of graders (constant for all hand-ins)
    N_Q = len(data.questions)
    
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
    mu_h = dict()
    tau_h = dict()
    mu_g = dict()
    tau_g = dict()
    T = defaultdict(dict)
    B = defaultdict(dict)

    # Draw from priors
    n_v = np.random.gamma(al_n, 1.0 / be_n)
    for h in data.handins.iterkeys():
        tau_h[h] = np.random.gamma(al_h, 1.0 / be_h)
        mu_h[h] = np.random.normal(ga_h, np.sqrt(1.0 / (la_h * tau_h[h])))
        for q in data.questions.iterkeys():
            T[h][q] = np.random.normal(mu_h[h], np.sqrt(1.0 / tau_h[h]))
    for g in data.graders.iterkeys():
        tau_g[g] = np.random.gamma(al_g, 1.0 / be_g)
        mu_g[g] = np.random.normal(ga_g, np.sqrt(1.0 / (la_g * tau_g[g])))
        for q in data.questions.iterkeys():
            B[g][q] = np.random.normal(mu_g[g], np.sqrt(1.0 / tau_g[g]))
    
    # Tracers initialising
    traces_n_v = list()
    traces_mu_h = defaultdict(list)
    traces_tau_h = defaultdict(list)
    traces_mu_g = defaultdict(list)
    traces_tau_g = defaultdict(list)
    traces_T = defaultdict(lambda: defaultdict(list))
    traces_B = defaultdict(lambda: defaultdict(list))

    tw = time.time()
    for step in range(burn_in + samples):
        print "\r%i out of %i" % ((step + 1), (burn_in + samples)),

        # Sample T
        for h, handin in data.handins.iteritems():
            for q, answers in handin.gradings.iteritems():
                n_gradings = len(answers.answers)
                sum_ = 0.0
                for g, val in answers.answers.iteritems():
                    sum_ = sum_ + val - B[str(g)][str(q)]
                v = n_v * n_gradings + tau_h[h]
                T[h][q] = np.random.normal((mu_h[h] * tau_h[h] +
                                            n_v * sum_) / v,
                                           np.sqrt(1.0 / v))
            
        # Sample B
        for g, grader in data.graders.iteritems():
            for q in data.questions.iterkeys():
                n_gradings = len(grader.handins)
                sum_ = 0.0
                for h in grader.handins:
                    if g in h.gradings[q].answers:
                        sum_ = sum_ + h.gradings[q].answers[g] - T[h.id][q]
                v = n_v * n_gradings + tau_g[g]
                B[g][q] = np.random.normal((mu_g[g] * tau_g[g] +
                                            n_v * sum_) / v,
                                           np.sqrt(1.0 / v))
        
        # Sample e
        sum_ = 0.0
        n_eval = 0
        for h, handin in data.handins.iteritems():
            for q, answers in handin.gradings.iteritems():
                for g, answer_val in answers.answers.iteritems():
                    n_eval = n_eval + 1
                    sum_ = sum_ + (answer_val - (T[h][q] + B[g][q]))**2
        n_v = np.random.gamma(al_n + 0.5 * n_eval, 1.0 / (be_n + 0.5 * sum_))

        # Sample u_q and t_q
        for h in data.handins.iterkeys():
            la_ = (la_h + N_Q)
            sum_q = 0.0
            for q in data.questions.iterkeys():
                sum_q = sum_q + T[h][q]
            mean_q = sum_q / N_Q
            sum_minus = 0.0
            for q in data.questions.iterkeys():
                sum_minus = sum_minus + (T[h][q] - mean_q)**2
            al_ = al_h + 0.5 * N_Q
            be_ = be_h + 0.5 * (N_Q * sum_minus + (N_Q * la_h *
                                (mean_q - ga_h)**2) / la_)
            tau_h[h] = np.random.gamma(al_, 1.0 / be_)
            mu_h[h] = np.random.normal((la_h * ga_h + sum_q) / la_,
                                       np.sqrt(1.0 / (la_ * tau_h[h])))

        # Sample mu_g and tau_g
        for g in data.graders.iterkeys():
            la_ = (la_g + N_Q)
            sum_q = 0.0
            for q in data.questions.iterkeys():
                sum_q = sum_q + B[g][q]
            mean_q = sum_q / N_Q
            sum_minus = 0.0
            for q in data.questions.iterkeys():
                sum_minus = sum_minus + (B[g][q] - mean_q)**2
            al_ = al_g + 0.5 * N_Q
            be_ = be_g + 0.5 * (N_Q * sum_minus +
                                (N_Q * la_g *
                                 (mean_q - ga_g)**2) / la_)
            tau_g[g] = np.random.gamma(al_, 1.0 / be_)
            mu_g[g] = np.random.normal((la_g * ga_g + sum_q) / la_,
                                       np.sqrt(1.0 / (la_ * tau_g[g])))
                        
        # Collect tracings
        if step > burn_in:
            traces_n_v.append(n_v)
            for h in data.handins.iterkeys():
                traces_mu_h[h].append(mu_h[h])
                traces_tau_h[h].append(tau_h[h])
                for q in data.questions.iterkeys():
                    traces_T[h][q].append(T[h][q])
            for g in data.graders.iterkeys():
                traces_mu_g[g].append(mu_g[g])
                traces_tau_g[g].append(tau_g[g])
                for q in data.questions.iterkeys():
                    traces_B[g][q].append(B[g][q])
                    
    print
    print "Wall time: %f" % (time.time() - tw)
    
    traces = {'n_v': traces_n_v,
              'mu_h': traces_mu_h,
              'tau_h': traces_tau_h,
              'mu_g': traces_mu_g,
              'tau_g': traces_tau_g,
              'T': traces_T,
              'B': traces_B}

    return traces

g_b = list()
g_e_b = list()
mh_b = list()
g_s = list()
g_e_s = list()
mh_s = list()
obs_s = list()

graders = 200
questions = 20
gradings = 5

print "Graders: %i" % graders
print "Gradings: %i" % gradings
print "Questions: %i" % questions 

for j in range(5):

    handins_data = list()
    graders_data = list()
    questions_data = list()
    answers_data = list()

    for i in xrange(questions):
        q = Question_('question_%i' % i)
        questions_data.append(q)

    for i in xrange(graders):
        g = Grader_('grader_%i' % i)
        g.set_bias(np.random.normal(0,tau_std(100)),np.random.gamma(50, 1.0 / 0.1))
        h = Handin_('handin_%i' % i, g)
        h.set_answers_scores()
        graders_data.append(g)
        handins_data.append(h)
        
    mock_data = Assignment_(handins_data,graders_data,questions_data,graders*gradings)
    mock_data.grade_mock_handins(gradings,np.random.gamma(50,1.0 / 0.1))
    
    print "Ext. gibbs"
    mock_gibbs_ext_result = gibbs_ext_model(mock_data,1000,0)

    print "Gibbs"
    mock_gibbs_result = gibbs_model(mock_data,1000,0)

    print "MH"
    mock_MH_result = MH_model(mock_data,1000,0)

    #print "PymC"
    #mock_MH_result = PyMC_peergrade_model(mock_data,1000,0)

    def MSE(true,estimated):
        return np.mean(map(lambda x : (x[0] - x[1])**2,zip(true,estimated)))

    def plot_handins(t,results,nth=1):
            
        labels = list()
        for (name,data) in results:
            labels.append(name)

        scores = list()
        for id, g in t.handins.iteritems():
            text = id
            val = list()
            for (name, data) in results:
                val.append(np.mean(data['mu_h'][id]))
            obs_scores = list()
            for grader in t.handins[id].graders:
                obs_scores.extend(map(lambda x: x[1], t.handins[id].get_grader_answers(grader.id)))
            scores.append((text, val, np.mean(data['mu_h'][id]), t.handins[id].mean, obs_scores))
        scores.sort(key=lambda x: x[3]) 

        true_score = map(lambda x : x[3],scores)

        for i in xrange(len(results)):
            x = map(lambda x : x[1][i],scores)
            #print "MSE %s: %f" % (labels[i], MSE(true_score,x))
            if labels[i] == "Ext. Gibbs":
                g_e_s.append(MSE(true_score,x))
            elif labels[i] == "MH":
                mh_s.append(MSE(true_score,x))
            elif labels[i] == "Gibbs":
                g_s.append(MSE(true_score,x))

        obs_score = map(lambda x : x[4],scores)
        #print "MSE %s: %f" % ("Observed average score", MSE(true_score,map(lambda x : np.mean(x),obs_score)))
        obs_s.append(MSE(true_score,map(lambda x : np.mean(x),obs_score)))


    def plot_bias(t,results,nth=1):
        
        labels = list()
        for (name,data) in results:
            labels.append(name)

        scores = list()
        for id, g in t.graders.iteritems():
            text = id
            text = text + ": %i" % len(t.graders[id].handins)
            val = list()
            for (name,data) in results:
                val.append(np.mean(data['mu_g'][id]))
            scores.append((text,val,np.mean(data['mu_g'][id]),t.graders[id].mean))
        scores.sort(key=lambda x:x[3])        
        
        true_bias = map(lambda x : x[3],scores)
        for i in xrange(len(results)):
            x = map(lambda x : x[1][i],scores)
            #print "MSE %s: %f" % (labels[i], MSE(true_bias,x))
            if labels[i] == "Ext. Gibbs":
                g_e_b.append(MSE(true_bias,x))
            elif labels[i] == "MH":
                mh_b.append(MSE(true_bias,x))
            elif labels[i] == "Gibbs":
                g_b.append(MSE(true_bias,x))

    print "Bias:"
    plot_bias(mock_data,[("Ext. Gibbs",mock_gibbs_ext_result),("Gibbs",mock_gibbs_result),("MH",mock_MH_result)],nth=4)
    print "Handin scores:"
    plot_handins(mock_data,[("Ext. Gibbs",mock_gibbs_ext_result),("Gibbs",mock_gibbs_result),("MH",mock_MH_result)],nth=4)

print "Bias:"
print "Ext. Gibbs [" + ",".join(map(str,g_e_b)) + "]"
print "Gibbs [" + ",".join(map(str,g_b)) + "]"
print "MH [" + ",".join(map(str,mh_b)) + "]"

print "Handin scores:"

print "Ext. Gibbs [" + ",".join(map(str,g_e_s)) + "]"
print "Gibbs [" + ",".join(map(str,g_s)) + "]"
print "MH [" + ",".join(map(str,mh_s)) + "]"
print "Obs [" + ",".join(map(str,obs_s)) + "]"