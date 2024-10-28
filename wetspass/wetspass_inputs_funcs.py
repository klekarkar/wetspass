
"""Description
This script prepares the input data for the WETSPASS model.
The input data includes gridded precipitation, temperature, wind, snow and potential evapotranspiration (PET) data.
For PET, the calculation of the PET is first done since it is not an 'observation'
The input data is prepared in the following steps:
"""
#%%
import os
import pandas as pd
import numpy as np
from pathlib import Path
from pandas import DataFrame
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import subprocess
import xarray as xr
from scipy.interpolate import griddata
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% precipitation data, humidity data,wind data
def gridded_input(df_path:Path, variable,start_date,resample_freq:str,min_x, max_x,min_y,max_y, gridsize):
    """
    This function prepares the input data for the WETSPASS model as arrays.

    Args:
    df_path (Path): Path to the CSV file containing met. variable data.
    variable (str): Name of the variable in the CSV file.
    resample_freq (str): Resampling frequency (e.g., 'M' for monthly, 'D' for daily).
    min_x, max_x, min_y, max_y (float): Minimum and maximum x and y coordinates of the grid.
    gridsize (float): Size of the grid cells.

    Output:
    target_grid_list (list): List of 2D arrays with the same shape as the grid. 
    Each 2D array contains the gridded data at each time step.
    """
    target_grid_list=[]   
    input_data = pd.read_csv(df_path, delim_whitespace=True, skiprows=3,header=None, names=['year', 'jday',variable]) #exclusively for files in swat format, otherwise use pd.read_csv(df_path) without skipping rows
    input_data['date'] = pd.to_datetime(input_data['year'].astype(int).astype(str) + input_data['jday'].astype(int).astype(str), format='%Y%j')
    input_data.set_index('date', inplace=True)
    input_data=input_data[start_date:]

    if variable=='rain':
        input_resampled=input_data.resample(resample_freq).sum().drop(columns=['year', 'jday'])
        #set zero values to a very low number
        input_resampled['rain'] = np.where(input_resampled['rain'] == 0, 0.01, input_resampled['rain']) #this avoids division by zero in the model
    else:
        input_resampled=input_data.resample(resample_freq).mean().drop(columns=['year', 'jday'])

    # create a 2D array with the same shape as the grid
    gridx = np.arange(min_x, max_x, gridsize,dtype=np.float64)
    gridy = np.arange(max_y,min_y,-gridsize, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(gridx, gridy)

    for i in range(len(input_resampled)):
        target_grid= np.full(x_grid.shape, np.round(input_resampled[variable].iloc[i],3),dtype=np.float32) #fill the grid with the value at each timestep  
        target_grid_list.append(target_grid)

    return target_grid_list

#*******************************************************************************************************************************************************


# %%grid temperature data
def gridded_temp(df_path:Path, start_date,resample_freq:str,min_x, max_x,min_y,max_y, gridsize):
    """
    Same as above but for temperature data
    """
    target_grid_list=[]
    input_data = pd.read_csv(df_path, delim_whitespace=True, skiprows=3,header=None, names=['year', 'jday','tmax','tmin'])
    input_data['date'] = pd.to_datetime(input_data['year'].astype(int).astype(str) + input_data['jday'].astype(int).astype(str), format='%Y%j')
    input_data.set_index('date', inplace=True)
    input_data['tmean']=(input_data['tmax']+input_data['tmin'])/2
    input_data=input_data[start_date:]
    input_resampled=input_data.loc[:,['tmean']].resample(resample_freq).mean()

    # create a 2D array with the same shape as the grid
    gridx = np.arange(min_x, max_x, gridsize,dtype=np.float64) #krigging performs operations on un
    gridy = np.arange(max_y,min_y,-gridsize, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(gridx, gridy)

    for i in range(len(input_resampled)):
        target_grid= np.full(x_grid.shape, np.round(input_resampled['tmean'].iloc[i],3),dtype=np.float32) #fill the grid with the value at each timestep
        target_grid_list.append(target_grid)

    return target_grid_list

#%%
###############################################################################################################################################################                         
def gridded_pet(df:DataFrame, variable,start_date,resample_freq:str,min_x, max_x,min_y,max_y, gridsize):
    """
    This function prepares the input data for the WETSPASS model as arrays.

    Args:
    df_path (Path): Path to the CSV file containing daily or pet.
    variable (str): Name of the variable in the CSV file.
    resample_freq (str): Resampling frequency (e.g., 'M' for monthly, 'D' for daily, 'Y' for annual).
    min_x, max_x, min_y, max_y (float): Minimum and maximum x and y coordinates of the grid.
    gridsize (float): Size of the grid cells.

    Output:
    target_grid_list (list): List of 2D arrays with the same shape as the grid. 
    Each 2D array contains the gridded data at each time step.
    """
    target_grid_list=[]   
    input_data=df[start_date:]
    input_resampled=input_data.loc[:,['pet']].resample(resample_freq).sum()

    # create a 2D array with the same shape as the grid
    gridx = np.arange(min_x, max_x, gridsize,dtype=np.float64)
    gridy = np.arange(max_y,min_y,-gridsize, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(gridx, gridy)

    for i in range(len(input_resampled)):
        target_grid= np.full(x_grid.shape, np.round(input_resampled[variable].iloc[i],3),dtype=np.float32) #fill the grid with the value at each timestep  
        target_grid_list.append(target_grid)

    return target_grid_list

#*******************************************************************************************************************************************************

# %%
"""
WRITE INPUT DATA ARRAYS TO ASCII GRID FILES
"""
def array_to_ascii_grid(array_list, start_date, var, units, gridx, gridy, crs,dest_folder):
    """
    This function writes the gridded input data to ASCII grid files.
    Args:
    array_list (list): List of 2D arrays with the same shape as the grid. Each 2D array contains the gridded data at each time step.
    dest_folder (str): Path to the folder where the ASCII grid files will be saved.
    var (str): Name of the variable (e.g., 'precipitation', 'temperature', 'wind', 'PET').
    """
    
    #Convert the NumPy array to an xarray DataArray
    dataset_list = []
    for array in array_list:
        xarray_var = xr.DataArray(array, coords=[gridy, gridx], dims=["lat", "lon"])
        var_dataset = xr.Dataset({var: xarray_var})
        dataset_list.append(var_dataset)


    # Create a time range from start to end with monthly frequency
    time_range = pd.date_range(start=start_date, periods=len(array_list), freq="M")


    # Assign the time range to the datasets
    for i, dataset in enumerate(dataset_list):
        dataset["time"] = time_range[i]

    #concatenate the datasets along the time dimension
    concatenated_dataset=xr.concat(dataset_list, dim="time")
    #assign global attributes to the dataset and specific attributes to the DataArray
    concatenated_dataset.attrs["crs"] = crs
    concatenated_dataset[var].attrs["units"] = units
    concatenated_dataset.attrs['description'] = f"Gridded {var} 10m resolution."
    concatenated_dataset.attrs['source'] = f"Gridded {var} from the Beggelbeek subcatchment, Flanders"
    concatenated_dataset.attrs['author'] = 'Katoria Lekarkar'

    ## Ensure destination folder exists
    outpath=os.path.join(dest_folder, var)
    os.makedirs(outpath, exist_ok=True)


    # Export to ASCII files in batches to minimize disk I/O
    for i in range(len(concatenated_dataset['time'])):
        # Construct file paths
        netcdf_file_path = os.path.join(outpath, f"{var}{i+1}.nc")
        ascii_file_path = os.path.join(outpath, f"{var}{i+1}.asc")

        # Save each time slice to a NetCDF file
        concatenated_dataset.isel(time=i).to_netcdf(netcdf_file_path)

        # Convert the NetCDF file to an ASCII grid file using gdal_translate
        subprocess.run(['gdal_translate', '-of', 'AAIGrid', '-ot', 'Float32', 
                        '-co', 'DECIMAL_PRECISION=4', '-a_nodata', '-9999', 
                        netcdf_file_path, ascii_file_path], check=True)

        # Remove the temporary NetCDF file to minimize disk usage
        os.remove(netcdf_file_path)
    return concatenated_dataset

# %% 
"""
POTENTIAL EVAPOTRANSPIRATION (PET) USING THE PENMAN-MONTEITH METHOD

This script computes the reference evapotranspiration (ETo) using the Penman-Monteith method.
"""

def compute_slope_of_vapor_pressure_curve(Tmean):
    """Computes the slope of the vapor pressure curve (delta) in kPa/C
    Args:
        T (float): temperature in degrees Celsius
    Returns:
        float: slope of the vapor pressure curve in kPa/C
    """
    delta= 4098 * (0.6108 * np.exp((17.27 * Tmean) / (Tmean + 237.3))) / (Tmean + 237.3) ** 2
    return delta


def compute_psychrometric_constant(elevation):

    """Computes the psychrometric constant (gamma) in kPa/C
    Args:
        elevation (float): elevation in meters
    Returns:
        float: psychrometric constant in kPa/C
    """
    #compute atmospheric pressure (P) in kPa
    P= 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26

    gamma=P * (0.665*10**-3)
    return gamma

def compute_saturation_vapor_pressure(Tmax, Tmin):
    """Computes the saturation vapor pressure (es) in kPa
    Args:
        Tmax (float):max. daily temperature in degrees Celsius
        Tmin (float):min. daily temperature in degrees Celsius
    Returns:
        float: saturation vapor pressure in kPa
    """
    e_Tmax=0.6108 * np.exp((17.27 * Tmax) / (Tmax + 237.3))
    e_Tmin=0.6108 * np.exp((17.27 * Tmin) / (Tmin + 237.3))
    es=(e_Tmax + e_Tmin) / 2
    return es


def dew_point_temperature(Tmean, RH):
    """Computes the dew point temperature (Td) in degrees Celsius
    Args:
        T (float): temperature in degrees Celsius
        RH (float): relative humidity in percentage
    Returns:
        float: dew point temperature in degrees Celsius
        Reference: Lawrence, M.G., 2005. The relationship between relative humidity and the dewpoint temperature in moist air:
        A simple conversion and applications. Bulletin of the American Meteorological Society, 86(2), 225-233.
    """
    a=17.625
    b=243.04
    td = b*(np.log(RH / 100) + a * Tmean / (b + Tmean)) / (a - np.log(RH / 100) - a * Tmean / (b + Tmean))
    return td

def vapor_pressure_deficit(Tmax, Tmin, Tmean, RHmean):
    """Computes the vapor pressure deficit (VPD) in kPa
    Args:
        Tmax (float): maximum temperature in degrees Celsius
        Tmin (float): minimum temperature in degrees Celsius
        RHmean (float): mean relative humidity in percentage
    Returns:
        float: vapor pressure deficit in kPa
    """
    es_tmax = compute_saturation_vapor_pressure(Tmax)
    es_tmin = compute_saturation_vapor_pressure(Tmin)
    es=(es_tmax + es_tmin) / 2
    ea = 0.6108*np.exp((dew_point_temperature(Tmean, RHmean)*17.27)/(dew_point_temperature(Tmean, RHmean)+237.3))

    return es - ea


def compute_2m_wind_speed(uz,z):
    """Converts wind speeds at different height to 2m wind speed (u2) in m/s
    Args:
        uz (float): measured wind speed at height z in m/s
    Returns:
        float: 2m wind speed in m/s
    """
    u2=uz*(4.87/(np.log(67.8*z-5.42))) #logarithmic wind profile
    return u2


def compute_net_radiation(latitude,day_of_year,elevation,Tmax,Tmin,Tmean, RHmean):
    #Reference: Allen, R.G., Pereira, L.S., Raes, D. and Smith, M., 1998. Crop evapotranspiration-Guidelines for computing crop water requirements-FAO Irrigation and drainage paper 56.
    #https://www.fao.org/4/X0490E/x0490e07.htm#radiation


    """Computes the net radiation (Rn) in MJ/m2/day
    Args:
        latitude (float): latitude in degrees
        day_of_year (int): day of the year
    Returns:
        float: net radiation in MJ/m2/day
    """
    #compute extraterrestrial radiation
    Gsc=0.082 #solar constant in MJ/m2/min
    dr=1+0.033*np.cos(2*np.pi*day_of_year/365) #inverse relative distance Earth-Sun
    phi=np.radians(latitude) #latitude degrees converted to radians (pi/180 converts degrees to radians) (positive for northern hemisphere)
    delta=0.409*np.sin((2*np.pi*day_of_year/365)-1.39) #solar decimation
    omega_ws=np.arccos(-np.tan(phi)*np.tan(delta)) #sunset hour angle in radians
    Ra=(24*60/np.pi)*Gsc*dr*(omega_ws*np.sin(phi)*np.sin(delta)+np.cos(phi)*np.cos(delta)*np.sin(omega_ws))

    #compute clear sky solar radiation (Rso) (MJ/m2/day)
    """Computes the clear sky solar radiation (Rso) in MJ/m2/day
    Args:
        Ra (MJ/m2/day)
        elevation (float): elevation in meters
    """
    Rso=(0.75+2*10**-5*elevation)*Ra

    #compute solar radiation (Rs) (MJ/m2/day)
    """Computes the solar radiation (Rs) in MJ/m2/day: 
    In absence of solar radiation data, Rs can be estimated using temperature data.
    Here the adjusted Hargreaves method is used to estimate Rs.
    The adjustment coefficient k_rs is set to 0.16 for inland areas and 0.19 for coastal areas.
    Args:   
        Tmax, Tmin (float): max, min temperature in degrees Celsius
        Ra (MJ/m2/day)
    """
    Rs=(0.16*np.sqrt(np.abs(Tmax-Tmin)))*Ra 

    #compute net shortwave radiation (Rns) (MJ/m2/day)
    """Computes the net shortwave radiation (Rns) in MJ/m2/day
    Args:
        Rso (MJ/m2/day)
        albedo (float): albedo
    """
    albedo=0.23
    Rns=(1-albedo)*Rs

    #compute net longwave radiation (Rnl) (MJ/m2/day)
    """Computes the net longwave radiation (Rnl) in MJ/m2/day
    Args:
        Tmin, Tmax (float): min, max temperature in degrees Celsius
        RHmean (float): mean relative humidity in percentage
        Rso (MJ/m2/day)
    """
    sigma=4.903*10**-9 #Stefan-Boltzmann constant in MJ/K^4/m^2/day
    ea=compute_saturation_vapor_pressure(dew_point_temperature(Tmean, RHmean))
    Rnl=sigma*(((Tmax+273.16)**4+(Tmin+273.16)**4)/2)*(0.34-0.14*np.sqrt(ea))*(1.35*Rns/Rso-0.35)

    #compute net radiation (Rn) (MJ/m2/day)
    Rn=Rns-Rnl

    return Rn


def compute_soil_heat_flux():

    """Computes the soil heat flux (G) in MJ/m2/day
    Args:
        Tmean (float): mean temperature in degrees Celsius
    Returns:
        float: soil heat flux in MJ/m2/day
    Since Soil Heat flux is small compared to Rn, it is often neglected in the Penman-Monteith method at daily time steps.
    
    For monthly time steps, G is often Gmonth,i = 0.14 (Tmonth,i - Tmonth,i-1 )
    """
    G=0
    return G

#-------------------------------------------------------------------------------------------------------------------------
def compute_penman_monteith_ETo(latitude, day_of_year, elevation, Tmax, Tmin, Tmean, RHmean, uz, z):
    """
    Computes the reference evapotranspiration (ET0) using the Penman-Monteith method.
    
    Args:
        latitude (float): Latitude in degrees.
        day_of_year (int): Day of the year.
        elevation (float): Elevation in meters.
        Tmax (float): Maximum daily temperature in degrees Celsius.
        Tmin (float): Minimum daily temperature in degrees Celsius.
        Tmean (float): Mean daily temperature in degrees Celsius.
        RHmean (float): Mean relative humidity in percent.
        uz (float): Wind speed at height z in m/s.
        z (float): Height at which wind speed is measured in meters.
        
    Returns:
        float: Reference evapotranspiration (ET0) in mm/day.
    """
    # Constants
    cp = 1.013e-3  # Specific heat of air at constant pressure (MJ/kg°C)
    epsilon = 0.622  # Ratio molecular weight of water vapour/dry air
    
    # Calculate the slope of vapor pressure curve
    delta = compute_slope_of_vapor_pressure_curve(Tmean)
    
    # Calculate saturation vapor pressure for Tmax and Tmin
    es_tmax = compute_saturation_vapor_pressure(Tmax)
    es_tmin = compute_saturation_vapor_pressure(Tmin)
    
    # Calculate mean saturation vapor pressure
    es = (es_tmax + es_tmin) / 2
    
    # Calculate actual vapor pressure using dew point temperature
    Td = dew_point_temperature(Tmean, RHmean)
    ea = compute_saturation_vapor_pressure(Td)
    
    # Calculate psychrometric constant
    gamma = compute_psychrometric_constant(elevation)
    
    # Compute net radiation
    Rn = compute_net_radiation(latitude, day_of_year, elevation, Tmax, Tmin, Tmean, RHmean)
    
    # Compute soil heat flux (assuming daytime conditions)
    G = compute_soil_heat_flux()
    
    # Convert wind speed from height z to 2 meters above ground level
    u2 = compute_2m_wind_speed(uz, z)
    
    # Penman-Monteith equation to calculate ETo. 0.408 converts MJ/m^2/day to mm/day (1/2.45)
    ETo_PM = ((0.408 * delta * (Rn - G)) + (gamma * (900 * (es - ea) * u2) / (Tmean + 273))) / (delta + gamma * (1 + 0.34 * u2))

    
    return ETo_PM

# Example usage:
#ETo = compute_penman_monteith_ETo(latitude=45, day_of_year=200, elevation=100, Tmax=5, Tmin=1, Tmean=3, RHmean=50, uz=2, z=10)
#print(f"Reference ETo: {ETo} mm/day")



#****************************************************************************************************************************************
#****************************************************************************************************************************************



# %% compute PET using the Hargreaves method
def hargreaves_ETo(Tmax, Tmin, Tmean, latitude, day_of_year):
    """Computes the reference evapotranspiration (ETo) using the Hargreaves method.
    Args:
        Tmax (float): Maximum temperature in degrees Celsius
        Tmin (float): Minimum temperature in degrees Celsius
        Tmean (float): Mean temperature in degrees Celsius
        latitude (float): Latitude in degrees
        day_of_year (int): Day of the year
    Returns:
        float: Reference evapotranspiration (ETo) in mm/day
    """
    # Constants
    Gsc = 0.082  # Solar constant in MJ/m2/min
    dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
    phi = np.radians(latitude)  # Latitude in radians
    delta = 0.409 * np.sin((2 * np.pi * day_of_year / 365) - 1.39)  # Solar declination in radians
    omega_ws = np.arccos(-np.tan(phi) * np.tan(delta))  # Sunset hour angle in radians
    Ra = (24 * 60 / np.pi) * Gsc * dr * (omega_ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(omega_ws))
    # Compute clear sky solar radiation (Rso) (MJ/m2/day)
    Rso = (0.75 + 2 * 10 ** -5 * 100) * Ra
    # Compute solar radiation (Rs) (MJ/m2/day)
    Rs = (0.16 * np.sqrt(np.abs(Tmax - Tmin))) * Ra
    # Compute reference evapotranspiration (ETo) (mm/day)
    ETo = 0.0023 * (Tmean + 17.8) * (Tmax - Tmin) ** 0.5 * Rs
    return ETo

#****************************************************************************************************************************************
#%%
"""
RAINY DAYS AND DEGREE DAYS PER MONTH
"""
#%%
def rainy_days_permonth(df_path:Path, start_date,resample_freq:str, outpath:Path):
    """This function calculates the number of rainy days per month and writes the output to a file.
    Args:
    df_path (Path): Path to the CSV file containing daily precipitation data.
    start_date (str): Start date in the format 'YYYY-MM-DD'.
    resample_freq (str): Resampling frequency (e.g., 'M' for monthly, 'D' for daily).
    outpath (Path): Path to the folder where the output file will be saved.
    Returns:
    DataFrame: Number of rainy days per month.
    """

    rain=pd.read_csv(df_path, delim_whitespace=True, skiprows=3,header=None, names=['year', 'jday','rain'])
    rain['date'] = pd.to_datetime(rain['year'].astype(int).astype(str) + rain['jday'].astype(int).astype(str), format='%Y%j')
    #set zero values to a very low number
    rain['rain'] = np.where(rain['rain'] == 0, 0.001, rain['rain']) #this avoids division by zero in the model
    rain.set_index('date', inplace=True)

    #start from the date where all inputs are available
    rain=rain[start_date:]
    rain['rainy_days'] = np.where(rain['rain'] > 1, 1, 0) #set 1 for rainy days and 0 for non-rainy days with a threshold of 1mm
    rainy_days=rain['rainy_days'].resample(resample_freq).sum().reset_index(drop=True)
    rainy_days.index += 1 #start index from 1

    #calculate average intensity of rainfall
    rain_month=rain['rain'].resample('M').sum().reset_index(drop=True)
    rain_month.index += 1 #start index from 1
    #intensity in mm/hr
    average_intensity=rain_month/rainy_days/24
    #get rid of inf values
    average_rain_intensity=average_intensity.replace([np.inf, -np.inf], np.nan).mean()


    rainy_days.to_csv(os.path.join(outpath,'RainyDaysPerMonth.TBL'), sep='\t',index_label='Code', header=['RainyDays'])

    return rainy_days, rain, average_rain_intensity

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

def degree_days_permonth(df_path:Path, start_date,resample_freq:str, outpath:Path):
    """This function calculates the degree days per month and writes the output to a file.
    Args:
    df_path (Path): Path to the CSV file containing daily temperature data.
    start_date (str): Start date in the format 'YYYY-MM-DD'.
    resample_freq (str): Resampling frequency (e.g., 'M' for monthly, 'D' for daily).
    outpath (Path): Path to the folder where the output file will be saved.
    Returns:
    DataFrame: Degree days per month.   
    """
    temp=pd.read_csv(df_path, delim_whitespace=True, skiprows=3,header=None, names=['year', 'jday','tmax','tmin'])
    temp['date'] = pd.to_datetime(temp['year'].astype(int).astype(str) + temp['jday'].astype(int).astype(str), format='%Y%j')
    temp.set_index('date', inplace=True)
    temp['tmean']=(temp['tmax']+temp['tmin'])/2
    temp=temp[start_date:]
    #caclulate degree days based on a characteristic temperature (mean daily temperature) above freezing point.
    temp['degree_days'] = np.where(temp['tmean'] > 0, 1, 0)
    degree_days=temp['degree_days'].resample(resample_freq).sum().reset_index(drop=True)
    degree_days.index += 1 #start index from 1

    degree_days.to_csv(os.path.join(outpath,'DegreeDaysPerMonth.TBL'), sep='\t',index_label='Code', header=['DegreeDays'])

    return degree_days
#***************************************************************************************************************************************

        

if __name__ == "__main__":
    # Example usage
    # Compute reference evapotranspiration (ETo) using the Penman-Monteith method
    ETo = compute_penman_monteith_ETo(latitude=45, day_of_year=200, elevation=100, Tmax=5, Tmin=1, Tmean=3, RHmean=50, uz=2, z=10)
    print(f"Reference ETo (Penman-Monteith): {ETo} mm/day")
    # Compute reference evapotranspiration (ETo) using the Hargreaves method
    ETo = hargreaves_ETo(Tmax=5, Tmin=1, Tmean=3,latitude=45, day_of_year=200)
    print(f"Reference ETo (Hargreaves): {ETo} mm/day")

#%%

  

# %%
