from random import randint
import numpy as np


class Radially_Constrained_Cluster(object):

    def __init__(self, data_to_cluster, n_seas, n_iter = 1000, learning_rate = 1, scheduling_factor = 1, min_len = 1, mode = 'single', n_ensemble = 1000, s_factor = 0.1):

        '''
            Compulsory parameters:
                -> data to cluster: time series with timesteps on first dimension and features on second
                -> n_seas: number of clusters

            Optional parameters:
                -> n_iter: number of iterations
                -> learning_rate: maximun number of day for stochastic breakpoints upgrade 
                -> scheduling_factor: factor for reducing learning_rate
                -> min_len: minimum length for bounded seasonal length
                -> mode: 'single' for single fit, 'ensemble_stochastic' for ensemble fit with stochastic parameters

            Experimental parameters:
                -> n_ensemble: number of ensemble fit
                -> s_factor: factor for stochastic parameters

        '''

        # Establishing the len of the serie
        self.len_serie = np.size(data_to_cluster,axis=0)
        self.data_to_cluster = data_to_cluster

        # Check parameter consistancy
        if self.len_serie/n_seas < min_len:
            raise ValueError(f'Cannot create {n_seas} season of {min_len} days. Please check your input parameters')
        else:
            self.n_seas = n_seas
            self.min_len = min_len

        # Setting parameters
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.scheduling_factor = scheduling_factor
        self.mode = mode
        self.s_factor = s_factor
        self.n_ensemble = n_ensemble


    def fit(self):

        '''
            Function for fitting the model and saving the results in the class.
            This functions manages the fitting mode (single vs ensemble) and calls the single fit function
            which contains the core of the algorithm.
        '''

        # Single mode fit: just one fit
        if self.mode == 'single':
            self.breakpoints, self.centroid_history, self.error_history, self.breakpoint_history, self.learningrate_history =  self.single_fit()


        ### THE ENSEMBLE MODE IS STILL SPERIMENTAL ###
        ### The randomization process of the parameters must be improved ###
        # Ensemble mode fit: multiple fit with stochastic parameters
        # if self.mode == 'ensemble_stochastic':

        #     # Defining list for metrics saving
        #     err = []
        #     bd = []

        #     # Main loop
        #     for j in range(self.n_ensemble):
                
        #         self.min_len = self.min_len + randint(-int(self.min_len/self.s_factor),int(self.min_len/self.s_factor))
        #         self.learning_rate = self.learning_rate + randint(-int(self.learning_rate/self.s_factor),int(self.learning_rate/self.s_factor))
        #         self.scheduling_factor = self.scheduling_factor + randint(-int(self.scheduling_factor/self.s_factor),int(self.scheduling_factor/self.s_factor))
        #         self.n_iter = self.n_iter + randint(-int(self.n_iter/self.s_factor),int(self.n_iter/self.s_factor))

        #         self.breakpoints, self.centroid_story , self.error_story =  self.single_fit()

        #         err.append(self.error_story[self.n_iter-1])
        #         bd.append(np.sort(self.breakpoints))
            
        #     self.breakpoints = np.int32(np.mean(bd,axis=0))



    def single_fit(self):

        # Defining list for metrics saving
        breakpoint_list = []
        centroid_list = []
        error_list = []
        learningrate_list = []
        

        # Main loop
        for j in range(self.n_iter):

            # Generating random starting breackpoints - equally distributed over time (firt iteration)
            if j == 0:
                upgrade, b = self.generate_starting_bpoints()

            # Randomly upgrading breakpoints in the range breakpoint +- learning rate (other iteration)
            else:
                upgrade, b = self.upgrade_breakpoints(b)

            # Generating index for each season
            idx = generate_season_idx(self.n_seas, b, self.len_serie)

            # Control on min season length - if false is skipped
            len_ok = check_season_len(self.n_seas, idx, self.min_len)

            # Case all season lengths are ok -> computing metrics
            if len_ok == True:
                centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx)

                breakpoint_list.append(b)
                centroid_list.append(centroids)
                error_list.append(np.sum(error))
                learningrate_list.append(self.learning_rate)

                # Skipping first iteration
                if j > 0:
                    # Checking if the breakpoints upgrade has improved the metrics
                    if error_list[j]>error_list[j-1]:
                        # If not downgrade breakpoints on last iteration
                        b = downgrade_breakpoints(self.n_seas, b, upgrade, self.len_serie)

                    # Scheduling learning rate for best minimun localization
                    elif (error_list[j-1] - error_list[j-2]) < 0 and self.scheduling_factor > 1 and self.learning_rate > 1:
                        self.learning_rate = schedule_learning_rate(self.learning_rate, self.scheduling_factor)

            # If there are too short seasons just pretend like nothing happend
            # Downgrading breakpoints to previous iteration
            else:
                b = downgrade_breakpoints(self.n_seas, b, upgrade, self.len_serie)
                idx = generate_season_idx(self.n_seas, b, self.len_serie)
                centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx)



        return np.sort(np.int32(b)), np.float64(centroid_list), np.float64(error_list), np.int32(breakpoint_list), np.int32(learningrate_list)
    



    def upgrade_breakpoints(self, old_b):

        upgrade = []
        new_b = []

        for k in range(self.n_seas):

            upgrade.append(randint(-self.learning_rate,self.learning_rate))
                
            new_b.append(old_b[k]+upgrade[k])

            if new_b[k]>self.len_serie-1:

                new_b[k]=new_b[k]-self.len_serie-1

            if new_b[k]<0:

                new_b[k]=self.len_serie-1+new_b[k]

        return upgrade, np.array(new_b)

    


    def generate_starting_bpoints(self):

        '''
            Function for generating starting breakpoints.  
        '''

        b_start = []
        upgrade = []

        # Core of breakpoints generation
        for i in range(self.n_seas):

            # If it's the first season 
            if i == 0:

                b_start.append(int((self.len_serie-1)/self.n_seas))
                upgrade.append(0)

            else:
            
                b_start.append(b_start[i-1]+int((self.len_serie-1)/self.n_seas))
                upgrade.append(0)

            if b_start[i] > self.len_serie-1:
                b_start[i] = b_start[i]-self.len_serie-1

        b_start = np.sort(b_start)

        return upgrade, b_start




    def get_prediction(self):

        # Converting breakpoints in a time series 
        prediction = np.zeros((self.len_serie,1))

        idx = generate_season_idx(self.n_seas, self.breakpoints, self.len_serie)

        for i in range(self.n_seas):
            prediction[idx[i]] = i

        return prediction


    def get_final_error(self):

        idx = generate_season_idx(self.n_seas, self.breakpoints, self.len_serie)

        centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx)

        return np.sum(error)
    
    
     
    def get_centroids(self):

        idx = generate_season_idx(self.n_seas, self.breakpoints, self.len_serie)

        centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx)

        return centroids
        

    def get_index(self):

        idx = generate_season_idx(self.n_seas, self.breakpoints, self.len_serie)

        return idx









def generate_season_idx(n_season, b, len_serie):

    idx = []


    if n_season == 1:

        idx.append(np.arange(0, len_serie, 1))

    

    else:



        for i in np.arange(-1, n_season-1,1):

            if b[i]>b[i+1]:

                idx_0 = np.arange(b[i], len_serie, 1)
                idx_1 = np.arange(0, b[i+1], 1)
                idx.append(np.concatenate((idx_0, idx_1), axis=None))
            

            else:
 
                idx.append(np.arange(b[i], b[i+1],1))


    return idx


def compute_metrics(n_season, data_to_cluster, idx):

    centroids = []
    error = []

    for i in range(n_season):
                
        centroids.append(np.nanmean(data_to_cluster[idx[i]], axis = 0))
        error.append(np.nansum(np.power(data_to_cluster[idx[i]]-centroids[i],2), axis = 0))

    return centroids, error





def downgrade_breakpoints(n_season, new_b, upgrade, len_serie):

    old_b = []

    for k in range(n_season):

        old_b.append(new_b[k]-upgrade[k])

        if old_b[k]>len_serie-1:

            old_b[k]=old_b[k]-len_serie-1

        if old_b[k]<0:

            old_b[k]=len_serie-1+old_b[k]

    return np.array(old_b)



def schedule_learning_rate(learning_rate, scheduling_factor):

    return np.int32(learning_rate/scheduling_factor)




def check_season_len(n_season, idx, min_len):

    len_ok = True

    for k in range(n_season):

        if len(idx[k])<min_len:

            len_ok = False

    return len_ok
