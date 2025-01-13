import numpy as np
from Likelihood import Logl


def walkergen(n, domain, vmn):
    x = np.zeros((n, 4))
    for i in range(0, n):
        for j in vmn:
            x[i][j] = np.random.uniform(domain[j][0], domain[j][1])
    return x


def NSalgo(posn, nwalkers, domain, fixed):
    Nsteps = 4000

    fixed = np.array(fixed)

    varparam = np.where(fixed == 0)[0]

    print(varparam)

    walkers = np.array(walkergen(nwalkers, domain, varparam))
    ll = np.zeros(nwalkers)
    for i in range(0, nwalkers):
        ll[i] = Logl(posn, walkers[i])

    # wt = 1/(nwalkers + 1)
    logwt = -np.log(nwalkers + 1)

    # these are the remainders in the jth iteration
    logZrem = np.log(0)

    # these are the Log Evidences for the respective models
    logZ = np.log(0)

    logp = min(ll) + logwt

    logZ = logsum(logZ, logp)

    for i in range(0, nwalkers):
        logZrem = logsum(ll[i], logZrem)

    logZrem = logZrem + logwt

    # initial stepsize
    step = 0

    # int index of the loop
    j = 1

    # tolerance value for stopping condition
    tol = np.log(10**(-4))

    # lowest likelihood in each iteration
    lstar = []

    # log weight
    logwt = -np.log(nwalkers + 1)

    # posterior avergae and posterior samples
    postavg = np.zeros(4)
    post = np.zeros(4)

    # posterior 2nd moment and posterior 2nd moment samples
    post2mavg = np.zeros(4)
    post2m = np.zeros(4)

    # information to calculate error
    H = 0

    # stopping ratio calculator
    R = logZrem - logZ

    if R < tol:
        print("term1 stop")
        print(logZ)

    while True:
        # log width contraction
        logwt = logwt + np.log(nwalkers / (nwalkers + 1))

        # log likelihood calculation
        logl = [Logl(posn, walkers[i]) for i in range(0, nwalkers)]

        # lowest likelihood of all the walkers
        lstar.append(min(logl))

        # walker index with minimum loglikelihood
        p = np.argmin(logl)

        # previous iterations log Evidence
        prevlogZ = logZ

        # previous iteration's H(information)
        prevH = H

        # log Z finding
        logZ = logsum(logZ, (lstar[-1] + logwt))

        # posterior sample
        post = (walkers[p]) * np.exp(lstar[-1]) * np.exp(logwt)

        # information calculation
        H += np.exp(logwt) * np.exp(lstar[-1]) * lstar[-1]

        print("Information")
        print(H)

        # posterior 2nd moment sample
        post2m = ((walkers[p]) ** 2) * np.exp(lstar[-1]) * np.exp(logwt)

        # posterior summation after each iteration
        postavg = postavg + post

        # posterior 2nd moment average
        post2mavg = post2mavg + post2m

        print("posterior")
        print(postavg)

        print("Evidence")
        print(logZ)

        # init Zremaining for the jth iteration
        logZrem = np.log(0)

        # calculate Zremaining for the jth iteration
        for i in range(0, nwalkers):
            logZrem = logsum(logl[i], logZrem)

        logZrem = logZrem + logwt

        print("Zrem =  " + str(logZrem))
        print("lowest likelihood = " + str(min(logl)))

        # ratio that determines the stopping ratio
        R = logZrem - logZ

        print("ratio")
        print(R)

        # break if stopping ratio is less than tolerance
        if R < tol:
            break

        # randomly select a walker to copy and random walk
        newwalkernumber = np.random.randint(0, nwalkers)

        # parameters of the randomly selected walker
        thetawalker = walkers[newwalkernumber]

        # walker replacement
        trace = MCMCwalker(posn, thetawalker,
                           lstar[-1], Nsteps, step, varparam)

        # step modulation in MCMC walker
        # step = trace[0]
        step = 1

        # replacing the worst walker with the copied walker that undergoes a
        # random walk
        walkers[p] = trace[1]

    # final evidence calulcation
    LogfinalZ = logsum(prevlogZ, logZrem)

    MLE = max(ll)

    # init posterior avg at the jmax iteration
    la = np.zeros(4)
    l2 = np.zeros(4)
    lh = 0
    Hfin = 0

    # calculation of posterior avg at the jmax iteration
    for i in range(0, nwalkers):
        la += walkers[i] * np.exp(ll[i])
        l2 += (walkers[i]**2) * np.exp(ll[i])
        lh += ll[i] * np.exp(ll[i])

    la *= np.exp(logwt)
    l2 *= np.exp(logwt)
    lh *= np.exp(logwt)

    # final posterior average
    postfin = postavg + la

    # final posterior 2 moment
    post2mfin = post2mavg + l2

    # dividing by the evidence of the model to find the posterior
    normpostfin = postfin / np.exp(LogfinalZ)

    # normalizing posterior 2nd moment
    normpost2mfin = post2mfin / np.exp(LogfinalZ)

    # calculating final information
    Hfin = prevH + lh
    Hfin = Hfin / np.exp(LogfinalZ) - LogfinalZ
    logZerr = np.sqrt(Hfin / nwalkers)

    # variance and error in the parameters
    varparam = normpost2mfin - normpostfin**2
    stddevparam = np.sqrt(varparam)

    # printing information
    print(
        "Posterior theta = " +
        str(normpostfin) +
        " Error = " +
        str(stddevparam))
    print("ln(Evidence) = " + str(LogfinalZ) + " +/- " + str(logZerr))
    print("Information = " + str(Hfin))

    return LogfinalZ, logZerr, normpostfin, stddevparam, Hfin, MLE


def logsum(a, b):
    z = max(a, b) + np.log(1 + np.exp(-np.abs(a - b)))
    return z


def MCMCwalker(x, walker, ll, N, step, VARPARAM):
    if step == 0:
        step = 1
    R = 0
    # intializing the walker distribution
    walker_new = np.zeros((N, 4))

    # init proposal function
    # proposal for new points in parameter space
    prop = np.zeros(4)

    # rejection count to determine new steps
    rejcount = 0

    # initial walker value is the intial walker point
    walker_new[0] = walker

    # VARPARAM = fixed[~1]

    # counter variable
    i = 1

    while i < N:

        # setting new values using the proposal function
        for j in VARPARAM:
            prop[j] = walker_new[i - 1][j] + np.random.normal(0, step)

        # log likelihood of the walker
        llwalker = Logl(x, prop)

        # moving around prior landscape
        if llwalker >= ll:
            walker_new[i] = prop
            ll = llwalker
        else:
            walker_new[i] = walker_new[i - 1]
            rejcount += 1
        i += 1

    # Rejection Ratio
    R = rejcount / N

    # new step
    step = min(step * np.exp(0.5 - R), 1)

    return step, walker_new[-1]
