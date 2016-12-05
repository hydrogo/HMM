"""
Hydrological modelling metrics (HMM)

Goodness of fit metrics for hydrological modeling tasks

Author: Georgy Ayzel (github.vom/hydrogo)

date: 05.12.2016

source: github.com/hydrogo/HMM
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def NS(obs, sim):
    """
    Nash-Sutcliff model efficinecy
    
    :paper: Nash, J. E., and J. V. Sutcliffe. 1970. River flow forecasting through conceptual models: Part 1. A discussion of principles. J. Hydrol. 10(3): 282-290.
    
        .. math::
         NSE = 1-\frac{\sum_{i=1}^{N}(obs_{i}-sim_{i})^2}{\sum_{i=1}^{N}(obs_{i}-\bar{obs})^2} 
    
    :observaliton: Observed data
    :type: numpy array
    
    :simulation: Simulated data
    :type: numpy array
    
    :return: Nash-Sutcliff model efficiency
    :rtype: float
    
    """
    numerator = np.sum( (obs - sim)**2 )
    
    denominator = np.sum( (obs - np.mean(obs))**2 )
    
    return 1 - numerator / denominator

def NSiq(obs, sim):
    """
    Nash-Sutcliff model efficinecy for inverse transformed flows
    
    :paper: Pushpalatha, R., Perrin, C., Moine, N. Le, Andréassian, V., 2012. A review of efficiency criteria suitable for evaluating low-flow simulations. J. Hydrol. 420–421, 171–182. doi:10.1016/j.jhydrol.2011.11.055
    
        .. math::
         NSE_{iq} = 1-\frac{\sum_{i=1}^{N}( \frac{1}{obs_{i} + \varepsilon} - \frac{1}{sim_{i} + \varepsilon})^2}{\sum_{i=1}^{N}(\frac{1}{obs_{i} + \varepsilon}-\bar{\frac{1}{obs_{i} + \varepsilon}})^2} 
    
    :observaliton: Observed data
    :type: numpy array
    
    :simulation: Simulated data
    :type: numpy array
    
    :return: Nash-Sutcliff model efficiency for inverse transformed flows
    :rtype: float
    
    """
    # add small constant for zero values avoiding
    obs = obs + 0.01 * obs.mean()
    sim = sim + 0.01 * sim.mean()
    
    # flows inverse transformation
    obs_iq = 1 / obs
    sim_iq = 1 / sim
    
    numerator = np.sum( (obs_iq - sim_iq)**2 )
    
    denominator = np.sum( ( obs_iq - np.mean(obs_iq) )**2 )
    
    return 1 - numerator / denominator

def KGE(obs, sim):
    """
    Modified Kling-Gupta Efficiency
    
    :paper: Kling, H., Fuchs, M., Paulin, M., 2012. Runoff conditions in the upper Danube basin under an ensemble of climate change scenarios. J. Hydrol. 424–425, 264–277. doi:10.1016/j.jhydrol.2012.01.011
    
        ..math::
            $$KGE' = 1 - \sqrt{ (r-1)^{2} + (\beta-1)^{2} + (\gamma-1)^{2}}$$

            $$\beta = \frac{\mu_{sim}}{\mu_{obs}} $$

            $$\gamma = \frac{ \sigma_{sim} / \mu_{sim} }{ \sigma_{obs} / \mu_{obs} }$$


            $\mu$ - mean runoff;

            $\sigma$ - standart deviation;

            $r$ - correlation coefficient.
       
    :observaliton: Observed data
    :type: numpy array
    
    :simulation: Simulated data
    :type: numpy array
    
    :return: Modified Kling-Gupta Efficiency
    :rtype: float
            
    """
    r     = np.corrcoef(obs, sim)[0,1]
    beta  = sim.mean() / obs.mean()
    gamma = ( sim.std() / sim.mean() ) / ( obs.std() / obs.mean() )
    
    kge   = 1 - np.sqrt( (r-1)**2 + (beta-1)**2 + (gamma-1)**2 )
        
    return kge

def bias(obs, sim):
    """
    Bias
    
        .. math::
          Bias=\frac{1}{N}\sum_{i=1}^{N}(obs_{i}-sim_{i})
    
    :observaliton: Observed data
    :type: numpy array
    
    :simulation: Simulated data
    :type: numpy array
    
    :return: Bias
    :rtype: float
    """
    return np.sum(obs - sim) / obs.size

def pbias(obs, sim):
    """
    Bias percentage
    
    :paper: Moriasi, D.N., Arnold, J.G., Van Liew, M.W., Binger, R.L., Harmel, R.D., Veith, T.L., 2007. Model evaluation guidelines for systematic quantification of accuracy in watershed simulations. Trans. ASABE 50, 885–900. doi:10.13031/2013.23153
    
        .. math::
          $$PBias= 100 * \frac{\sum_{i=1}^{N}(obs_{i}-sim_{i})}{\sum_{i=1}^{N}(obs_{i})} \%$$
    
    :observaliton: Observed data
    :type: numpy array
    
    :simulation: Simulated data
    :type: numpy array
    
    :return: PBias
    :rtype: float
    """
    return 100 * ( np.sum(obs - sim) / np.sum(obs) )

def fdc(series, interval=[0, 1], plot=False):    
    """
    FDC (flow duration curve) coordinates calculation
    
    source:
    http://pubs.usgs.gov/wsp/1542a/report.pdf
    
    :series: flow series
    :type: numpy array
    
    :interval: probability interval
    :type: list with floats - min and max of probability
    :default: [0, 1] - whole curve
    
    :plot: if True - plot simple FDC
    :type: boolean
    :default: False
    
    :return: probabilities, ascending sorted series
    :rtype: numpy arrays
    """  
    
    series = np.sort(series)
    
    # ranks data from smallest to largest
    ranks = spstats.rankdata(series)

    # reverses rank order
    ranks = ranks[::-1]
    
    # calculate probability of each rank
    prob = np.array([ ( ranks[i] / (len(series)+1) ) for i in range( len(series) ) ])
    
    mask = (interval[0] < prob) & (prob < interval[1]) 
    
    if plot == True:
        plt.plot(prob[mask], series[mask])
    
    return prob[mask], series[mask]

def FHV5(obs, sim):
    """
    FHV5 is a bias in FDC high-segment volume (0-5%)
    
    :paper: Yilmaz, K.K., Gupta, H. V., Wagener, T., 2008. A process-based diagnostic approach to model evaluation: Application to the NWS distributed hydrologic model. Water Resour. Res. 44, n/a-n/a. doi:10.1029/2007WR006716
    
        .. math::
          $$\Delta FHV = 100 * \frac{\sum_{h=1}^{H}(sim_{h}-obs_{h})}{\sum_{h=1}^{H}(obs_{h})} \%$$
          $h = 1, 2, ...H$ - flow indices for flows with exceedance probabilities lower than 0.05
    
    :observaliton: Observed data
    :type: numpy array
    
    :simulation: Simulated data
    :type: numpy array
    
    :return: FHV5
    :rtype: float
    """
    # implement fdc() function for flows slicing from 0-5%
    obs_sorted = fdc(obs, interval=[0, 0.05])[1]
    sim_sorted = fdc(sim, interval=[0, 0.05])[1]
    
    # lenght of sorted flow arrays may not match
    # we need to cut arrays for minimum length of both
    min_lenght = min(obs_sorted.size, sim_sorted.size)
    
    obs_sorted = obs_sorted[: min_lenght]
    sim_sorted = sim_sorted[: min_lenght]
    
    FHV5 = 100 * ( np.sum( sim_sorted - obs_sorted ) / np.sum(obs_sorted) )
    
    return FHV5

def FLV70(obs, sim):
    """
    FLV70 is a bias in FDC low-segment volume (> 70%)
    
    :paper: Yilmaz, K.K., Gupta, H. V., Wagener, T., 2008. A process-based diagnostic approach to model evaluation: Application to the NWS distributed hydrologic model. Water Resour. Res. 44, n/a-n/a. doi:10.1029/2007WR006716
    
        .. math::
          $$\Delta FLV = -100 * \frac{ \sum_{l=1}^{L}[ log(sim_{l}) - log(sim_{L}) ] - \sum_{l=1}^{L}[ log(obs_{l}) - log(obs_{L}) ] }{ \sum_{l=1}^{L}[ log(obs_{l}) - log(obs_{L}) ] } \%$$

          $l = 1, 2, ...L$ - index of the flow value located within the low-flow segment (0.7–1.0 flow exceedance probabilities) of the flow duration curve, L being the index of the minimum flow.
    
    :observaliton: Observed data
    :type: numpy array
    
    :simulation: Simulated data
    :type: numpy array
    
    :return: FLV70
    :rtype: float
    
    """
    # implement fdc() function for flows slicing from 70-100%
    obs_sorted = fdc(obs, interval=[0.7, 1])[1]
    sim_sorted = fdc(sim, interval=[0.7, 1])[1]
    
    # lenght of sorted flow arrays may not match
    # we need to cut arrays for minimum length of both
    min_lenght = min(obs_sorted.size, sim_sorted.size)
    
    obs_sorted = obs_sorted[: min_lenght]
    sim_sorted = sim_sorted[: min_lenght]
    
    # log-transform flow values
    obs_log = np.log(obs_sorted)
    sim_log = np.log(sim_sorted)
    
    numerator = np.sum(sim_log - sim_log[0]) - np.sum(obs_log - obs_log[0])
    
    denominator = np.sum(obs_log - obs_log[0])
        
    FLV70 = -100 * (numerator / denominator)
    
    return FLV70

def FLV90(obs, sim):
    """
    FLV90 is a bias in FDC low-segment volume (> 90%)
    
    :paper: Yilmaz, K.K., Gupta, H. V., Wagener, T., 2008. A process-based diagnostic approach to model evaluation: Application to the NWS distributed hydrologic model. Water Resour. Res. 44, n/a-n/a. doi:10.1029/2007WR006716
    
        .. math::
          $$\Delta FLV = -100 * \frac{ \sum_{l=1}^{L}[ log(sim_{l}) - log(sim_{L}) ] - \sum_{l=1}^{L}[ log(obs_{l}) - log(obs_{L}) ] }{ \sum_{l=1}^{L}[ log(obs_{l}) - log(obs_{L}) ] } \%$$

          $l = 1, 2, ...L$ - index of the flow value located within the low-flow segment (0.7–1.0 flow exceedance probabilities) of the flow duration curve, L being the index of the minimum flow.
    
    :observaliton: Observed data
    :type: numpy array
    
    :simulation: Simulated data
    :type: numpy array
    
    :return: FLV70
    :rtype: float
    
    """
    # implement fdc() function for flows slicing from 70-100%
    obs_sorted = fdc(obs, interval=[0.9, 1])[1]
    sim_sorted = fdc(sim, interval=[0.9, 1])[1]
    
    # lenght of sorted flow arrays may not match
    # we need to cut arrays for minimum length of both
    min_lenght = min(obs_sorted.size, sim_sorted.size)
    
    obs_sorted = obs_sorted[: min_lenght]
    sim_sorted = sim_sorted[: min_lenght]
    
    # log-transform flow values
    obs_log = np.log(obs_sorted)
    sim_log = np.log(sim_sorted)
    
    numerator = np.sum(sim_log - sim_log[0]) - np.sum(obs_log - obs_log[0])
    
    denominator = np.sum(obs_log - obs_log[0])
        
    FLV90 = -100 * (numerator / denominator)
    
    return FLV90
