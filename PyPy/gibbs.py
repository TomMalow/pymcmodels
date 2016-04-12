import sys
from collections import defaultdict
import numpy as np
import time
# Add the application folder to the path
sys.path.insert(0,'../../Peergrade/peergrade/')
import application.model as data_model



class Grader(object):
    def __init__(self, name):
        self.name = name
        self.handins = list()
        
    def add_handin(self, handin):
        self.handins.append(handin)

class Handin:
    def __init__(self,title,owner):
        self.title = title
        self.owner = owner
        self.gradeings = dict()
        self.graders = list()
    
    def add_grader(self,grader):
        self.graders.append(grader)
        
    def add_gradeing(self,grader,value):
        self.gradeings[grader.name] = value

        
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


def user_name(user_id):
    user = data_model.User.objects.get(id=user_id)
    return user.name

def question_max_value(question):
    if question.question_type == "boolean":
        return 1
    elif question.question_type == "numerical":
        if question.numericalAnswers:
            max_value = max(map(int,question.numericalAnswers.keys()))
            return max_value
        else:
            return 5

def answer_value(answer):
    if answer.numerical_answer != None:
        return answer.numerical_answer / float(question_max_value(answer.question))
    if answer.boolean_answer != None:
        return answer.boolean_answer / float(question_max_value(answer.question))

def score_handin(report_grade):
    answers = data_model.Answer.objects(report_grade=report_grade)
    handin_n = 0.0
    handin_acc = 0.0
    for answer in answers:
        if answer.text_answer == None:
            handin_acc = handin_acc + answer_value(answer)
            handin_n = handin_n + 1.0
    return handin_acc / handin_n

def fetch_assignment_data(ass_obj):
    '''
    Takes an course and assignment data model object and transforms it into populated Assignment object
    Only student or all?
    '''

    ## Find all graders
    graders = dict()
    for student_ in ass_obj.course.students:
        graders[str(student_.id)] = Grader(str(student_.id))
    
    ## Find all handins
    handins = dict()
    for handin_ in data_model.Handin.objects(assignment=ass_obj):
        handins[str(handin_.id)] = Handin(str(handin_.id),str(handin_.submitter.id))
        
    ## Find all handins graders have graded and vice versa
    n_gradings = 0
    for handin_ in data_model.Handin.objects(assignment=ass_obj):
        for grade in data_model.ReportGrade.objects(handin=handin_,state='ANSWERED'):
            n_gradings = n_gradings + 1
            
            # Needed if TA or Professor have graded reports as they are not initialy part of it
            if str(grade.giver.id) not in graders:
                graders[str(grade.giver.id)] = Grader(str(grade.giver.id))

            handins[str(handin_.id)].add_grader(graders[str(grade.giver.id)])
            handins[str(handin_.id)].add_gradeing(graders[str(grade.giver.id)],score_handin(grade))
            
    ## update reference in graders
    for handin in handins.itervalues():
        for grader in handin.graders:
            grader.add_handin(handin)

    return Assignment(handins.itervalues(),graders.itervalues(),n_gradings)

def fetch_data(obj):

    res_c = Course()
        
    if type(obj).__name__ == "Course":
        assignments_d = data_model.Assignment.objects(course=obj)
        for assignment_d in assignments_d:
            res_c.add_assignment(fetch_assignment_data(assignment_d))
    elif type(obj).__name__ == "Assignment":
        res_c.add_assignment(fetch_assignment_data(obj))
        
    return res_c

def gibbs_model(data):
    
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
    e = np.random.gamma(al_e,1.0/be_e)
    for h in data.handins.iterkeys():
        t_h[h] = np.random.gamma(al_h,1.0/be_h)
        u_h[h] = np.random.normal(ga_h,np.sqrt(1.0/(la_g * t_h[h])))
        T[h] = np.random.normal(u_h[h],np.sqrt(1.0/t_h[h]))
    for g in data.graders.iterkeys():
        t_g[g] = np.random.gamma(al_g,1.0/be_g)
        u_g[g] = np.random.normal(ga_g,np.sqrt(1.0/(la_g * t_g[g])))
        B[g] = np.random.normal(u_g[g],np.sqrt(1.0/t_g[g]))

    # Gibbs sampling
    
    burn_in = 1000  # warm-up steps
    samples = 5000 # Gibbs sampling steps
    
    # Tracers initialising
    acc_e = 0
    acc_u_h = defaultdict(lambda : 0)
    acc_t_h = defaultdict(lambda : 0)
    acc_u_g = defaultdict(lambda : 0)
    acc_t_g = defaultdict(lambda : 0)
    acc_T = defaultdict(lambda : 0)
    acc_B = defaultdict(lambda : 0)

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
            T[h] = np.random.normal((u_h[h]*t_h[h]+e*sum_)/v,np.sqrt(1/v))
            
        # Sample B
        for g, grader in data.graders.iteritems():
            n_gradings = len(grader.handins)
            sum_ = 0.0
            for h in grader.handins:
                sum_ = sum_ + h.gradeings[g] - T[h.title]
            v = e*n_gradings+t_g[g]
            B[g] = np.random.normal((u_g[g]*t_g[g]+e*sum_)/v,np.sqrt(1/v))
        
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
            al_ = al_h + 0.5 * la_h + 0.5 * np.square(T[h]-u_h[h])
            be_ = be_h + 0.5 + 0.5 * 1.0
#            al_ = al_h+0.5
#            be_ = be_h+0.5*((la_h*np.square(T[h]-ga_h))/la_)
            t_h[h] = np.random.gamma(al_,1.0/be_)
            u_h[h] = np.random.normal((la_h*ga_h+T[h])/la_,np.sqrt(1.0/(la_*t_h[h])))

        # Sample u_g and t_g
        for g in data.graders.iterkeys():
            la_ = (la_g+1.0)
            al_ = al_g + 0.5 * la_g + 0.5 * np.square(B[g]-u_g[g])
            be_ = be_g + 0.5 + 0.5 * 1.0
#            al_ = al_g+0.5
#            be_ = be_g+0.5*((la_g*np.square(B[g]-ga_g))/la_)
            t_g[g] = np.random.gamma(al_,1.0/be_)
            u_g[g] = np.random.normal((la_g*ga_g+B[g])/la_,np.sqrt(1.0/(t_g[g])))
            
        # Collect tracings
        if r > burn_in:
            acc_e = acc_e + e
            for h in data.handins.iterkeys():
                acc_u_h[h] = acc_u_h[h] + u_h[h]
                acc_t_h[h] = acc_t_h[h] + t_h[h]
                acc_T[h] = acc_T[h]+ T[h]
            for g in data.graders.iterkeys():
                acc_u_g[g] = acc_u_g[g]+ u_g[g]
                acc_t_g[g] = acc_t_g[g] + t_g[g]
                acc_B[g] = acc_B[g] + B[g]    
    
    acc_e = acc_e / float(samples)
    for h in data.handins.iterkeys():
        acc_u_h[h] = acc_u_h[h] / float(samples)
        acc_t_h[h] = acc_t_h[h] / float(samples)
        acc_T[h] = acc_T[h] / float(samples)
    for g in data.graders.iterkeys():
        acc_u_g[g] = acc_u_g[g] / float(samples)
        acc_t_g[g] = acc_t_g[g] / float(samples)
        acc_B[g] = acc_B[g] / float(samples)
    
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

###

a1 = data_model.Assignment.objects.get(title="Your Choice of Subject")

a1_data = fetch_data(a1)

a1_result = gibbs_model(a1_data)
