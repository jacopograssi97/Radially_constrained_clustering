import random
from random import randint
import numpy as np
from tqdm import tqdm

random.setstate((3,
    (2468570525,
    44967195,
    2667364560,
    2449893699,
    1652692239,
    766678126,
    273175325,
    1513475390,
    2407048223,
    2326550691,
    3055735416,
    2487780036,
    476975371,
    81632736,
    1598452444,
    3338301038,
    3898475993,
    1749546629,
    4084786842,
    949316744,
    2086501466,
    4175211502,
    3792229788,
    1718685282,
    2499662139,
    4222931543,
    3063257123,
    910424605,
    1400804300,
    830603822,
    3216023045,
    2756927633,
    3684278863,
    3724968901,
    332416530,
    52016619,
    2751489098,
    1877715228,
    1932382287,
    3281876149,
    3597828351,
    330629843,
    142483984,
    1379430288,
    83784318,
    2266112133,
    1736800492,
    3746267091,
    2610492607,
    2079803227,
    3463890091,
    615297649,
    2445958069,
    138783768,
    741209753,
    3721915402,
    2027708325,
    4005341927,
    2093884772,
    119215273,
    551524651,
    3739622759,
    3782730527,
    404717681,
    321534867,
    1286801508,
    1706479953,
    2882329788,
    1029701930,
    2373551443,
    3296995744,
    468358352,
    746091816,
    4096927057,
    641317208,
    2423816852,
    662051236,
    1347945045,
    744683282,
    3532103569,
    3323996770,
    674188488,
    2147579353,
    4002509157,
    1635774310,
    2870381986,
    1633495405,
    3350196287,
    225215418,
    1170120648,
    915993856,
    814856433,
    196876581,
    2157558451,
    3897838842,
    3150173549,
    626324766,
    2067876245,
    2163845165,
    4042368565,
    1376677108,
    1262248675,
    2205442378,
    3993334766,
    9743238,
    2593325684,
    2920379669,
    1534455130,
    3818766181,
    931649853,
    2158376649,
    3577176492,
    4105269980,
    2743411340,
    2855498512,
    3468322221,
    4289135738,
    3070378031,
    130878110,
    2012459331,
    3649976437,
    1132601439,
    747682378,
    48846564,
    660000069,
    1790312343,
    3727890972,
    1155723235,
    1514429407,
    1230076367,
    1013715474,
    4196577359,
    1320124222,
    2614278628,
    1297893158,
    4083753327,
    2352894470,
    947894400,
    2642100948,
    1169889630,
    1286436482,
    3306394082,
    3164045139,
    1094362406,
    809487105,
    2843373296,
    2280653556,
    2080861721,
    1562856334,
    994764831,
    4181417961,
    1060980731,
    2404272427,
    3309777776,
    1336994281,
    634755732,
    3631638369,
    1391515368,
    1418228798,
    4257897983,
    2054225289,
    567832856,
    1330177904,
    2462727694,
    814045371,
    2591348022,
    743574337,
    2789138291,
    2041853854,
    894395601,
    2564448893,
    2991512555,
    661658788,
    4244382938,
    592840949,
    4198784705,
    4208381264,
    1027548464,
    1699297713,
    3507187687,
    4228784501,
    3944198753,
    393010807,
    1855658975,
    2650303920,
    837948699,
    3219332495,
    2923291683,
    2860126530,
    3856051376,
    2249134764,
    165767879,
    2468337443,
    1781864276,
    2657744714,
    35449830,
    828146831,
    117482919,
    3433429317,
    1819066727,
    710883018,
    3107854316,
    3076257894,
    928245986,
    1936492070,
    1083117887,
    4108585320,
    313911202,
    235106869,
    3091059945,
    905889358,
    259789608,
    3447145250,
    988142971,
    2178196317,
    859662840,
    1908755715,
    1247277970,
    1481142601,
    819671330,
    2548134350,
    1495134650,
    4034870622,
    2814194974,
    2761218509,
    2977430738,
    614006212,
    981226091,
    413177493,
    3471336991,
    2131872665,
    4009914404,
    612529023,
    378607496,
    2988973248,
    2418016553,
    3050435072,
    3405173865,
    239315520,
    553425169,
    2806326921,
    2194625577,
    1297818883,
    557367713,
    1339678305,
    625637250,
    3007124173,
    1403416408,
    963253146,
    557613038,
    2995233521,
    1599272606,
    2877491804,
    3025784937,
    3444226192,
    3778689225,
    2511282536,
    2036290414,
    3663672933,
    870613663,
    3288722796,
    1883286129,
    2240711678,
    1598432647,
    1653428643,
    1037288789,
    3417332711,
    632265342,
    2992319607,
    2229992519,
    2627094451,
    2902395192,
    1798625598,
    1888821172,
    2928617356,
    2806510607,
    2169745473,
    3263400237,
    477483472,
    2684152104,
    2047416023,
    1061764082,
    3888197689,
    3665203944,
    3081648115,
    1585188167,
    979304208,
    3283599107,
    515443754,
    3528859579,
    2646985622,
    1179116369,
    3174096483,
    3622666293,
    1094110660,
    982532210,
    3915875056,
    3442760653,
    2482674618,
    3543561277,
    4242258297,
    1883210421,
    198934262,
    3881993543,
    3270985024,
    3814018289,
    3842198594,
    3180274062,
    349497396,
    2056365044,
    3662991668,
    2471767104,
    2872942732,
    1154111690,
    3142477833,
    2062459812,
    3422415124,
    352502659,
    3206123932,
    769305078,
    1282348479,
    3011976512,
    1592394005,
    976424517,
    3257644548,
    2159244792,
    3015546726,
    1321951765,
    1457127034,
    1008018749,
    1340492242,
    3250697729,
    1439525819,
    2116389080,
    3128629141,
    3912463512,
    2778908372,
    5179345,
    2764285036,
    4013718511,
    76636421,
    2440399146,
    4124147582,
    1565329027,
    2314846721,
    2825257189,
    554997050,
    2676063690,
    3230428478,
    4066464853,
    3785792675,
    3491102306,
    1012514472,
    710423760,
    1104362914,
    1402276434,
    870434098,
    64327618,
    245834932,
    4099459452,
    3866904251,
    2240453378,
    1724463324,
    1330601334,
    3433676187,
    829295067,
    3806454686,
    950099493,
    4293362446,
    594307004,
    79190971,
    2311908688,
    54171305,
    62487414,
    3504337811,
    2771970015,
    1836590151,
    2595431378,
    3416341100,
    3453307109,
    1174988285,
    2852396363,
    346848325,
    2368812712,
    226406421,
    3941277996,
    3989222844,
    3009299209,
    1702732764,
    2598609657,
    3925497101,
    331397553,
    388553728,
    3553027581,
    2831176302,
    1171547784,
    2429194224,
    1919275555,
    2943364212,
    392528745,
    2077320491,
    416107366,
    3505919650,
    2641506636,
    3367202201,
    2496764115,
    223919825,
    271108961,
    2545966472,
    1316212361,
    3137675020,
    49774935,
    2744430138,
    3230926645,
    1183214045,
    1795720081,
    3453588112,
    891938360,
    4144344690,
    2777301904,
    1995233055,
    3359734316,
    896930090,
    3330969507,
    3223398016,
    1321717194,
    4215086939,
    3506673919,
    100418703,
    2598322782,
    1873905913,
    1698737593,
    1965703533,
    60435064,
    1751428005,
    1152971074,
    3618663090,
    3158488445,
    3727477430,
    657970680,
    1511931134,
    1717050987,
    310598970,
    2234372010,
    1017571582,
    4084110079,
    2305036871,
    4254307802,
    2941750258,
    2165051637,
    1472622743,
    2543351527,
    1796705211,
    2214600371,
    686749318,
    4022876929,
    2100068217,
    3727699398,
    3217299548,
    275738892,
    78573358,
    2500678662,
    2944914056,
    1277909152,
    2318080503,
    3799903604,
    2033312710,
    1430582106,
    2681053359,
    427226790,
    4052010686,
    1405513990,
    283355798,
    2154582023,
    3237342184,
    2326232545,
    3053750987,
    3682467274,
    4258665988,
    1693455081,
    3276042809,
    1890575484,
    3321173492,
    1435919955,
    372744468,
    2288550928,
    130181578,
    464432903,
    2644098717,
    850876397,
    366381834,
    1912868480,
    4114884255,
    2076074274,
    2025154398,
    3191648339,
    1180631776,
    1821926123,
    142706752,
    3139028750,
    3108622860,
    1876156978,
    3356317510,
    3260050869,
    2334989316,
    747109268,
    4016280193,
    2897996881,
    2994915453,
    803723030,
    1933605890,
    3104516246,
    533383945,
    701195023,
    2592103620,
    1356972692,
    1491149426,
    4160117465,
    3960597945,
    2567279869,
    1374045353,
    3117232482,
    139766291,
    2589485771,
    1707073928,
    3210823559,
    537281128,
    10518971,
    1901873126,
    2898897661,
    573642982,
    760245815,
    3807024923,
    2334167321,
    1211114995,
    3530176240,
    1229318785,
    3602144670,
    1250553934,
    1010089880,
    2172233573,
    2688964066,
    3758094780,
    2941802101,
    1581001398,
    3746782544,
    2917164021,
    252667418,
    1150188760,
    3542252877,
    1389159379,
    1906599979,
    3288259755,
    778740684,
    358910446,
    26153786,
    443928973,
    1407665083,
    298990169,
    3405562703,
    504530202,
    3362938768,
    1086122129,
    3588952012,
    177358838,
    1668686040,
    1788441005,
    2920778456,
    3450590302,
    1707705043,
    3940504028,
    1650147200,
    2144853533,
    429939140,
    2060161875,
    226622212,
    1271791848,
    3603087696,
    48155551,
    966813043,
    984177119,
    3033759521,
    3492815891,
    2391190442,
    3575857178,
    3965974952,
    459455113,
    59851712,
    416034666,
    1727702234,
    3862955095,
    2038677741,
    405912737,
    3651584525,
    1433865433,
    4162114042,
    319642522,
    120211088,
    3610217925,
    1667950605,
    284010502,
    2536690859,
    1757606927,
    98163371,
    1298766898,
    2843598018,
    2749694903,
    3031345259,
    2633279512,
    2812045979,
    34084905,
    2989448216,
    3311204930,
    763257776,
    747261640,
    127287928,
    326017657,
    2610204813,
    3746483709,
    1345625337,
    76875111,
    1840566970,
    4008707741,
    1079217633,
    1),
    None))

class Radially_Constrained_Cluster(object):

    def __init__(self, data_to_cluster, n_seas, n_iter = 1000, learning_rate = 1, scheduling_factor = 1, min_len = 1, mode = 'single', n_ensemble = 1000, s_factor = 0.1):

        '''
            Mandatory parameters:
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
            self.breakpoints, self.centroid_history, self.error_history, self.breakpoint_history, self.learningrate_history, self.prediction_history =  self.single_fit()


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
        prediction_history = []
        breakpoint_list = []
        centroid_list = []
        error_list = []
        learningrate_list = []
        

        # Main loop
        for j in tqdm(range(self.n_iter)):

            # Generating random starting breakpoints - equally distributed over time (firt iteration)
            if j == 0:
                upgrade, b = self.generate_starting_bpoints()

            # Randomly upgrading breakpoints in the range breakpoint +- learning rate (other iteration)
            else:
                upgrade, b = self.upgrade_breakpoints(b)

            # Generating index for each season
            idx = self.generate_season_idx(b)

            # Control on min season length - if false is skipped
            len_ok = self.check_season_len(idx)

            # Case all season lengths are ok -> computing metrics
            if len_ok == True:

                # Storing current bp
                self.breakpoints = b
                breakpoint_list.append(self.breakpoints)

                self.prediction = self.get_prediction()
                prediction_history.append(self.prediction)

                centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx)
                centroid_list.append(centroids)
                error_list.append(np.sum(error))
                learningrate_list.append(self.learning_rate)

                # Skipping first iteration
                if j > 0:
                    # Checking if the breakpoints upgrade has improved the metrics
                    if error_list[-1]>error_list[-2]:
                        # If not downgrade breakpoints on last iteration
                        b = downgrade_breakpoints(self.n_seas, b, upgrade, self.len_serie)

                    # # Scheduling learning rate for best minimun localization
                    # elif (error_list[-1] - error_list[-2]) < 0 and self.scheduling_factor > 1 and self.learning_rate > 1:
                    #     self.learning_rate = schedule_learning_rate(self.learning_rate, self.scheduling_factor)

            # If there are too short seasons just pretend like nothing happend
            # Downgrading breakpoints to previous iteration
            else:
                b = downgrade_breakpoints(self.n_seas, b, upgrade, self.len_serie)
                idx = self.generate_season_idx(b)
                centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx)

        return np.sort(np.int32(b)), np.float64(centroid_list), np.float64(error_list), np.int32(breakpoint_list), np.int32(learningrate_list), np.int32(prediction_history)
    



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

        idx = self.generate_season_idx(self.breakpoints)

        for i in range(self.n_seas):
            prediction[idx[i]] = i

        return prediction


    def get_final_error(self):

        idx = self.generate_season_idx(self.breakpoints)

        centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx)

        return np.sum(error)
    
    
     
    def get_centroids(self):

        idx = self.generate_season_idx(self.breakpoints)

        centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx)

        return centroids
        

    def get_index(self):

        idx = self.generate_season_idx(self.breakpoints)

        return idx









    def generate_season_idx(self, b):

        idx = []

        if self.n_seas == 1:
            idx.append(np.arange(0, self.len_serie, 1))

        else:
            for i in np.arange(-1, self.n_seas-1,1):
                if b[i]>b[i+1]:
                    idx_0 = np.arange(b[i], self.len_serie, 1)
                    idx_1 = np.arange(0, b[i+1], 1)
                    idx.append(np.concatenate((idx_0, idx_1), axis=None))

                else:
                    idx.append(np.arange(b[i], b[i+1],1))

        return idx
    


    def check_season_len(self, idx):

        len_ok = True

        for k in range(self.n_seas):

            if len(idx[k])<self.min_len:

                len_ok = False

        return len_ok


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





