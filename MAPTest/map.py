from collections import defaultdict
import pymc as pm
import numpy as np
import time

def random_limit(distribution):
    val = distribution.random()
    while val > 1 or val < 0:
        val = distribution.random()
    return val


class grader(object):
    def __init__(self, name, bias_mean, bias_tau):
        self.name = name
        self.handins = list()
        self.bias_mean = bias_mean
        self.bias_tau = bias_tau

    def add_handin(self, handin):
        self.handins.append(handin)

    def grade_handins(self):
        for handin in self.handins:
            B = pm.Normal('B_generator', self.bias_mean, self.bias_tau)
            handin.add_gradeing(self, B.random())


class handin:
    def __init__(self, title, owner, true_value, precision):
        self.title = title
        self.owner = owner
        self.gradeings = dict()
        self.graders = list()
        self.true_val = true_value
        self.precision = precision

    def add_grader(self, grader):
        self.graders.append(grader)

    def add_gradeing(self, grader, bias):
        obs = pm.Normal('obs_generator', self.true_val+bias, self.precision)
        self.gradeings[grader.name] = random_limit(obs)


class assignment(object):

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
        sorted_l = sorted(self.handins.values(), key=lambda x: len(x.graders))
#        i = int(random.uniform(0, len(sorted_l)))
        i = 0
        handin = sorted_l[i]
#        while(handin.owner.name == grader.name):
        while handin in grader.handins or (handin.owner.name == grader.name):

            i += 1
#            i = int(random.uniform(0, len(sorted_l)))
            handin = sorted_l[i]
        return handin

    def grade_handins(self, n_handins):
        # Distribute handins
        for i in xrange(0, n_handins):
            for grader in self.graders.itervalues():
                h = self.find_ungraded_handin(grader)
                h.add_grader(grader)
                grader.add_handin(h)

        # grade handins
        for grader in self.graders.itervalues():
            grader.grade_handins()

def generate_data(assignments, graders, handins):
    T_mu = pm.Normal('T_mu_generator', 0.5, 25)
    T_tau = pm.Gamma('T_tau_generator', 10, 0.1)
    B_mu = pm.Normal('B_mu_generator', 0, 100)
    B_tau = pm.Gamma('B_tau_generator', 50, 0.1)

    handins_data = list()
    graders_data = list()

    for i in xrange(0, graders):
        g = grader('grader_%i' % i, B_mu.random(), B_tau.random())
        t_mu = random_limit(T_mu)
        h = handin('handin_%i' % i, g, t_mu, T_tau.random())
        graders_data.append(g)
        handins_data.append(h)

    assignment_data = assignment(handins_data, graders_data)
    assignment_data.grade_handins(handins)
    return_values = [assignment_data]

    for a in xrange(1, assignments):

        handins_data_e = list()
        grader_min = graders * a
        grader_max = grader_min + graders
        for i in xrange(grader_min, grader_max):
            t_mu = random_limit(T_mu)
            h = handin('handin_%i' % i, graders_data[i-grader_min], t_mu, T_tau.random())
            handins_data_e.append(h)
            
        assignment_data_e = assignment(handins_data_e, graders_data)
        assignment_data_e.grade_handins(handins)
        return_values.append(assignment_data_e)

    return return_values

def Model(data):
    N_H = len(data)
    
    # Bias
    T_tau = dict()
    T_mu = dict()
    B_mu = dict()
    B_tau = dict()
    O = list()

    for h in range(0, N_H):
        h_id = data[h].title
        scores = data[h].gradeings.items()
        
        N_G = len(scores)
        T_mu[h_id] = pm.Normal('T_mu_%s' % str(h_id), 0.5, 25)
        T_tau[h_id] = pm.Gamma('T_tau_%s' % str(h_id), 10, 0.1)
        
        for g in range(0, N_G):
            (g_id, val) = scores[g]
            
            if g_id not in B_mu:
                B_mu[g_id] = pm.Normal('B_mu_%s' % str(g_id), 0, 100)
            if g_id not in B_tau:
                B_tau[g_id] = pm.Gamma('B_tau_%s' % str(g_id), 50, 0.1)
                
            O.append(pm.Normal('O_%(h)i_%(g)i' % {'h': h, 'g': g}, mu=T_mu[h_id] + B_mu[g_id], tau=T_tau[h_id] + B_tau[g_id], observed=True, value=val))
               
    collection = [pm.Container(T_mu),
                  pm.Container(T_tau),
                  pm.Container(B_mu),
                  pm.Container(B_tau),
                  pm.Container(O)]
    
    model = pm.Model(collection)
    return model


def execute_model_map(model, samples):
    map_ = pm.MAP(model)
    map_.fit()
    mcmc = pm.MCMC(model)
    mcmc.sample(samples)#, progress_bar=False)
    return mcmc


def execute_model_no_map(model, samples, burn):
    mcmc = pm.MCMC(model)
    mcmc.sample(samples, burn=burn)#, progress_bar=False)
    return mcmc


def build_mcmc(model_, values):
    values_ = list()
    for ass in values:
        values_.extend(ass.handins.values())
    return model_(values_)

def find_bias(assignments, mcmc, f):
    bias = list()
    # All assignments should have the same graders
    for g in assignments[0].graders.keys():
        value = 0
        if f == 'var':
            value = np.mean(mcmc.trace('B_tau_%s' % str(g))[:])
        elif f == 'mean':
            value = np.mean(mcmc.trace('B_mu_%s' % str(g))[:])
        bias.append((value, g))
    return bias


def find_T(assignments, mcmc, f):
    T = list()
    for ass in assignments:
        for h in ass.handins.keys():
            value = 0
            if f == 'var':
                value = np.mean(mcmc.trace('T_tau_%s' % str(h))[:])
            elif f == 'mean':
                value = np.mean(mcmc.trace('T_mu_%s' % str(h))[:])
            T.append((value, h))
    return T


def find_MSE(assignment_data, mcmc_handins, find, func='mean'):
    found = find(assignment_data, mcmc_handins, func)

    # Generate dict of the found values in each run for each grader
    compared = defaultdict(list)
    collected = list()
    collected = found[:]

    for (value, _id) in collected:
        compared[_id].append(value)

    sorted_list = list()
    if find.func_name == "find_bias":
        for _id, g in assignment_data[0].graders.iteritems():
            if func == "mean":
                sorted_list.append((_id, g.bias_mean))
            else:
                sorted_list.append((_id, g.bias_tau))
    else:
        for ass in assignment_data:
            for _id, h in ass.handins.iteritems():
                if func == "mean":
                    sorted_list.append((_id, h.true_val))
                else:
                    sorted_list.append((_id, h.precision))

    sorted_list.sort(key=lambda x: x[1])

    true_values = list()
    score_values = list()
    MS_val = list()
    for (_id, value) in sorted_list:
        MS_val.append(np.mean(compared[_id]))
        score_values.append(compared[_id])
        true_values.append(value)

    MSE_M = sum(map(lambda x: (float(x[1]) - float(x[0])) ** 2, zip(true_values, MS_val))) / len(true_values)
    return MSE_M

# assignments graders, gradings per grader
setup = [(1, 10, 5),
         (2, 10, 5),
         (1, 20, 10),
         (2, 20, 10)]

for i, (assignments, graders, gradings) in enumerate(setup):
    print "Running setup %i" % (i+1)
    print str(assignments) + " assignments, " + str(graders) + " graders, " + str(gradings) + " gradings per grader"

    print ""

    print "Generating data"
    tc = time.clock()
    tw = time.time()
    data = generate_data(assignments, graders, gradings)
    print "Process time: %f" % (time.clock() - tc)
    print "Wall time: %f" % (time.time() - tw)

    print "Building model"
    tc = time.clock()
    tw = time.time()
    mcmc = build_mcmc(Model, data)
    print "Process time: %f" % (time.clock() - tc)
    print "Wall time: %f" % (time.time() - tw)

    print "Executing model with no MAP"
    tc = time.clock()
    tw = time.time()
    result_no_map = execute_model_no_map(mcmc, 10000, 2500)
    print "Process time: %f" % (time.clock() - tc)
    print "Wall time: %f" % (time.time() - tw)

    tc = time.clock()
    tw = time.time()
    print "Executing model with MAP"
    result_map = execute_model_map(mcmc, 2500)
    print "Process time: %f" % (time.clock() - tc)
    print "Wall time: %f" % (time.time() - tw)

    print ""
    print "Calculating Mean Square Error"
    print "No MAP used: " + str(find_MSE(data, result_no_map, find_bias))
    print "MAP used: " + str(find_MSE(data, result_map, find_bias))
    print ""
