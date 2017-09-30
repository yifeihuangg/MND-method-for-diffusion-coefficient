def get_error(pos, estimations, true_D, KDE_True, Drift_True):
    """
    make errorbar plot, and calculate root mean square error, RMSE of diffusion coefficient estimations.

    Parameters
    -----------
    pos: array
       an array of positions, where the estimation is made.
    estimations: list of arrays(matrix)
        each list is an estimation of D corresponding to certain positions, each coloum corresponding to the same position but different estimations from different models.
    true_D: array
        true D values.
    KDE_True: TRUE/FALSE
        if True, then we use KDE method to estimate. if False, then we use MND method.
    Drift_True: TRUE/FALSE
        if True, we use KDE method considering about the drift term bias correction. if False, we use simple KDE method without correction.

    Returns
    -----------
    rmse: array
       rmse of D as fxn of position x.
    """
    # calculate mean for each col, that's the mean for each position
    mean = np.mean(estimations,axis = 0)
    # calculate sd for each col, that's the sd for each position
    sd = np.std(estimations,axis = 0)
    #plot error bar and true
    plt.figure()
    plt.scatter(pos,true_D,label = 'true')
    if KDE_True == 'TRUE':
        if Drift_True == 'TRUE':
            plt.errorbar(pos,mean,yerr = sd, linestyle = 'None', marker = '^', label = 'KDE_w_drift')
        elif Drift_True == 'FALSE':
            plt.errorbar(pos,mean,yerr = sd, linestyle = 'None', marker = '^', label = 'KDE_wo_drift')
        else:
            print("Please check the Drift_True parameter, TRUE/FALSE")
    elif KDE_True == 'FALSE':
        plt.errorbar(pos,mean,yerr = sd, linestyle = 'None', marker = '^', label = 'MND')
    else:
        print("Please check the KDE_True parameter, TRUE/FALSE")
    plt.xlabel('positions(x)')
    plt.ylabel('Diffusion coefficient(D)')
    plt.legend(loc = 2)
    if KDE_True == 'TRUE':
        if Drift_True == 'TRUE':
            plt.title('Error bar plot of D KDE method w/ drift')
            plt.savefig('error_bar_KDE_w_drift')
        elif Drift_True == 'FALSE':
            plt.title('Error bar plot of D KDE method w/o drift')
            plt.savefig('error_bar_KDE_wo_drift')
    elif KDE_True == 'FALSE':
        plt.title('Error bar plot of D MND method')
        plt.savefig('error_bar_MND')
    plt.show()
    # calculate RMSE
    mse = mean_squared_error(true_D, estimations,multioutput = 'raw_values' )
    rmse = np.sqrt(mse)
    plt.figure()
    plt.plot(pos,rmse)
    plt.xlabel('positions(x)')
    plt.ylabel('RMSE(root mean square error)')
    plt.legend()
    if KDE_True == 'TRUE':
        if Drift_True == 'TRUE':
            plt.title('RMSE of D KDE method w/ drift')
            plt.savefig('error_bar_KDE_w_drift')
        elif Drift_True == 'FALSE':
            plt.title('RMSE of D KDE method w/o drift')
            plt.savefig('error_bar_KDE_wo_drift')
    elif KDE_True == 'FALSE':
        plt.title('RMSE of D MND method')
        plt.savefig('error_bar_MND')
    plt.show()

    return rmse



def get_error_entirespace(pos, estimations, true_D, KDE_True, Drift_True, H, KT = 1.0):
    """
    Calculate error over entire space for D estimations.

    Parameters
    -----------
    pos: array
       an array of positions, where the estimation is made.
    estimations: list of arrays(matrix)
        each list is an estimation of D corresponding to certain positions, each coloum corresponding to the same position but different estimations from different models.
    true_D: array
        true D values.
    KDE_True: TRUE/FALSE
        if True, then we use KDE method to estimate. if False, then we use MND method.
    Drift_True: TRUE/FALSE
        if True, we use KDE method considering about the drift term bias correction. if False, we use simple KDE method without correction.
    H: array
        Hamiltonian of the system, in this system is the potential energy with respect to positions.
    KT: float
        Boltzmann constant * temprature. Default is 1.0.

    Returns
    -----------
    e_ent_space: array
       error over entire space of D as fxn of position x.
    """
    beta = 1/KT
    error = np.zeros_like(estimations)
    sum = 0
    # Calculate the denominator(sum of exp(beta*H(x)))
    for k in xrange(len(estimations[0])):
        sum += np.exp(-(beta*H[k]))
    # Calculate the error term for every position in every estimation
    for i in xrange(len(estimations)):
        for k in xrange(len(estimations[0])):
            error[i][k] = (((estimations[i][k]-true_D[i][k])**2)*(np.exp(-(beta*H[k]))))/sum
    # To get the error over entire space, calculate the coloum average of matrix (avg error with same position)
    error_ent_space = np.mean(error,axis = 0)

    plt.figure()
    plt.plot(pos,error_ent_space)
    plt.xlabel('positions(x)')
    plt.ylabel('error over entire space')
    plt.legend()
    if KDE_True == 'TRUE':
        if Drift_True == 'TRUE':
            plt.title('Error_ent_space of D KDE method w/ drift')
            plt.savefig('error_ent_space_KDE_w_drift')
        elif Drift_True == 'FALSE':
            plt.title('Error_ent_space of D KDE method w/o drift')
            plt.savefig('error_ent_space_KDE_wo_drift')
    elif KDE_True == 'FALSE':
        plt.title('error_ent_space of D MND method')
        plt.savefig('error_ent_space_MND')
    plt.show()

    return error_ent_space

def get_error_tstate(tstate, tstate_est, true_D_tstate, KDE_True, Drift_True):
    """
    make errorbar plot, and root mean square error plot, calculate RMSE of D for transition state.

    Parameters
    -----------
    tstate: array
       an array of transition state positions.
    tstate_est: list of arrays(matrix)
        each list is an estimation of D at transition state, each coloum corresponding to the same position but different estimation result from different models.
    true_D_tstate: array
        true D values at transition state.
    KDE_True: TRUE/FALSE
        if True, then we use KDE method to estimate. if False, then we use MND method.
    Drift_True: TRUE/FALSE
        if True, we use KDE method considering about the drift term bias correction. if False, we use simple KDE method without correction.

    Returns
    -----------
    rmse_tstate: array
       rmse of D at transition state.
    """
    # calculate mean for each col, that's the mean for each t_state
    mean = np.mean(tstate_est,axis = 0)
    # calculate sd for each col, that's the sd for each t_state
    sd = np.std(tstate_est,axis = 0)
    #plot error bar and true_D_tstate
    plt.figure()
    plt.scatter(tstate,true_D_tstate,label = 'true')
    if KDE_True == 'TRUE':
        if Drift_True == 'TRUE':
            plt.errorbar(tstate,mean,yerr = sd, linestyle = 'None', marker = '^', label = 'KDE_w_drift')
        elif Drift_True == 'FALSE':
            plt.errorbar(tstate,mean,yerr = sd, linestyle = 'None', marker = '^', label = 'KDE_wo_drift')
        else:
            print("Please check the Drift_True parameter, TRUE/FALSE")
    elif KDE_True == 'FALSE':
        plt.errorbar(tstate,mean,yerr = sd, linestyle = 'None', marker = '^', label = 'MND')
    else:
        print("Please check the KDE_True parameter, TRUE/FALSE")
    plt.xlabel('positions(x)')
    plt.ylabel('Diffusion coefficient(D)')
    plt.legend(loc = 2)
    if KDE_True == 'TRUE':
        if Drift_True == 'TRUE':
            plt.title('Transition Error bar plot KDE method w/ drift')
            plt.savefig('transition_error_bar_KDE_w_drift')
        elif Drift_True == 'FALSE':
            plt.title('Transition Error bar plot KDE method w/o drift')
            plt.savefig('transition_error_bar_KDE_wo_drift')
    elif KDE_True == 'FALSE':
        plt.title('Transition Error bar plot of D MND method')
        plt.savefig('transition_error_bar_MND')
    plt.show()
    # calculate RMSE
    mse_tstate = mean_squared_error(true_D_tstate, tstate_est,multioutput = 'raw_values' )
    rmse_tstate = np.sqrt(mse_tstate)
    plt.figure()
    plt.plot(tstate,rmse_tstate)
    plt.xlabel('transition state position')
    plt.ylabel('RMSE(root mean square error)')
    plt.legend()
    if KDE_True == 'TRUE':
        if Drift_True == 'TRUE':
            plt.title('transition state RMSE of D KDE method w/ drift')
            plt.savefig('transition_error_bar_KDE_w_drift')
        elif Drift_True == 'FALSE':
            plt.title('transition state RMSE of D KDE method w/o drift')
            plt.savefig('transition_error_bar_KDE_wo_drift')
    elif KDE_True == 'FALSE':
        plt.title('transition state RMSE of D MND method')
        plt.savefig('transition_error_bar_MND')
    plt.show()

    return rmse_tstate


    def get_U(x):
        """
        Returns the potential energy of a function that is a quartic double well of form (x^4-2x^2).

        Parameters
        ----------
        x : float
            The location in the one-dimensional coordinate.
        a : float
            The prefactor for the potential.

        Returns
        -------
        U : float
            Value of the potential.
        """
        xsq = x*x
        print xsq, xsq*xsq, 2*xsq
        return xsq*xsq-2.*xsq


    def get_D(x):
        """
        Returns the value of the diffusion function at x.

        Parameters
        ----------
        x : float
            The location in the one-dimensional coordinate.

        Returns
        -------
        D : float
            Value of the diffusion function.
        """
        return np.sin(x)*0.5+1

    def main():
        # read in MND estimations into MND_est
        for i in xrange(100):
            with open('D_MND_%d.txt' %(i+1),"r") as est:
                MND_est = []
                for line in est:
                    MND_est.append(line)

        MND_est = np.array(MND_est)

        # get positions
        hist_edges = np.linspace(-2.,2.,101)
        # positions used to estimate true D, use middle point of each histogram bin
        pos = np.zeros(len(hist_edges)-1)
        for m in xrange(len(hist_edges)-1):
            pos[m] = (hist_edges[m]+hist_edges[m+1])/2

        # get true D
        D = get_D(pos)
        true_D = np.array([D[:,0]]*len(MND_est)

        # get Hamiltonian
        hami = get_U(pos)

        # get KT
        KT = 1.0

        # get transition state estimation, position and true_D_tstate
        tstate = np.array([-1,0,1])
        tstate_est = np.zeros_like(tstate)
        tstate_est[0] = (MND_est[24]+MND_est[25])/2
        tstate_est[1] = (MND_est[49]+MND_est[50])/2
        tstate_est[2] = (MND_est[74]+MND_est[75])/2
        true_D_tstate = get_D(tstate)

        RMSE_MND = get_error(pos, MND_est, true_D, KDE_True = 'FALSE', Drift_True = 'FALSE')
        err_ent_space_MND = get_error_entirespace(pos, MND_est, true_D, KDE_True = 'FALSE', Drift_True = 'FALSE', H = hami, KT = KT)
        err_tstate_MND = get_error_tstate(tstate, tstate_est, true_D_tstate, KDE_True = 'FALSE', Drift_True = 'FALSE')
