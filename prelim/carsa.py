import xarray as xr
import numpy as np

def FFDI(precip, rh, tmax, wmax, time_dim='time'):
    """ Returns the McArthur Forest Fire Danger Index following the formula provided in Dowdy (2018):
        FFDI = D ** 0.987 * exp (0.0338 * T - 0.0345 * H + 0.0234 * W + 0.243147)
        
        Parameters
        ----------
        precip : xarray DataArray
            Daily total precipitation [mm]. This is used to estimate the drought factor, D, as the 20-day accumulated rainfall
            scaled to lie between 0 and 10, with larger values indicating less precipitation.
        rh : xarray DataArray
            Daily max relative humidity at 2m [%] (or similar, depending on data availability). Richardson et al. uses mid-afternoon
            relative humidity at 2 m, Squire et al. (2021) uses daily mean relative humidity at 1000 hPa. This is used as H in the
            above equation.
        tmax : xarray DataArray
            Daily max 2 m temperature [deg C].
        wmax : xarray DataArray
            Daily max 10 m wind speed [km/h] (or similar, depending on data availability). Squire et al. (2021) uses daily mean wind speed.
            
        Returns
        -------
        FFDI : xarray DataArray
            Array containing the FFDI.
    """
    
    p20 = precip.rolling({time_dim: 20}).sum()
    
    D = -10 * ((p20 - p20.min(time_dim)) / (p20.max(time_dim) - p20.min(time_dim))) + 10
    
    return D ** 0.987 * np.exp(0.0338 * tmax - 0.0345 * rh + 0.0234 * wmax + 0.243147)