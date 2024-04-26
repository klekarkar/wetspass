#%%import libraries
import os
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import pykrige.kriging_tools as kt
import xarray as xr
import subprocess
from pathlib import Path
from typing import List

# %%
#----------------------------------------------------------------

def ordinary_kriging(z_df, source_coordinates_df, min_x,max_x,min_y,max_y,gridsize,variogram_model):
    """
    Performs ordinary kriging on the given data and returns the interpolated values
    z_df: DataFrame containing the data (x, y, z): x and y are the coordinates and z is the value to be interpolated
    source_coordinates: DataFrame containing the coordinates of the data points
    min_x, min_y, max_x, max_y: minimum and maximum x and y coordinates of the study area
    z-values: z-values of the data points
    variogram_model: variogram model to be used for kriging
    nlags: number of lags to be used for the variogram model
    weight: weight of the variogram model
    """
    for i in range(len(z_df)): #iterate through each timestep
        z_array=z_df.iloc[i,:].to_numpy().reshape(-1,1) #reshape converts the array to a single column so that each row can be associated with a coordinate

        '''sensor coordinates. The first two columns are the x and y coordinates. The rest are not needed, hence left out
        sensor coords is a dataframe containing the coordinates of piezometers'''
        data_coords=source_coordinates_df.iloc[:,:2].to_numpy() 

        # The data: x,y,z, where z (gw data) is the value to be interpolated.
        valid_indices = ~np.isnan(z_array) #remove locations with nan values. it causes an error in the interpolation
        gw_monthly_array_clean = z_array[valid_indices]

        #extract the coordinates of the valid indices. piezometer with nan values are removed
        #make sure both array shapes are compatible
        data_coords_clean = data_coords.reshape(-1,2)[valid_indices[:, 0]] #valid_indices is a 2D boolean array. Select the first column to filter the coordinates to get rid of nan measurements.

        #stack the data and the coordinates
        z_data_stack = np.hstack((data_coords_clean, gw_monthly_array_clean.reshape(-1, 1)))

        # Generate grid points based on the size of the mask and the resolution of the DEM
        #Gridpoints correspond to the bounds of the DEM/study area to interpolate the groundwater data (BE Lambert Coordinate system). 5 is the resolution of the DEM.
        gridx = np.arange(min_x, max_x, gridsize,dtype=np.float64) #krigging performs operations on un
        gridy = np.arange(max_y,min_y,-gridsize, dtype=np.float64)

        # Perform Ordinary Kriging. I grabbed the code from the pykrige documentation
        '''https://geostat-framework.readthedocs.io/projects/pykrige/en/latest/examples/00_ordinary.html'''
        # Create the ordinary kriging object. Required inputs are the X-coordinates of the data points, the Y-coordinates of the data points, and the Z-values of the data points.
        OK = OrdinaryKriging(
            z_data_stack[:, 0],  # x-values
            z_data_stack[:, 1],  # y-values
            z_data_stack[:, 2],  # z-values (groundwater data)
            variogram_model=variogram_model,  # Adjust variogram model as needed
            verbose=False,
            enable_plotting=False,
        )

        # Execute kriging to generate the interpolated values on the grid
        z, ss = OK.execute("grid", gridx, gridy)

        #save the interpolated data and the grid spacing
        interpolated_data=z
        return interpolated_data,ss, z_data_stack

        #write data to ascii grid
        #kt.write_asc_grid(gridx, gridy, interpolated_gw[::-1], filename=f"idw_gw_ts_{5}.asc", style=2) #invert the array

#--------------------------------------------------------------------------------------------------------------------------------------
# %% 
"""create interpolation data arrays"""
def create_interpolation_data(spatial_data: pd.DataFrame, 
                       sensor_coords: pd.DataFrame)-> List[np.ndarray]:
    """
    Prepares spatial data for interpolation by associating groundwater measurements
    with their spatial coordinates.
    
    Parameters:
    - spatial_data: DataFrame containing the groundwater data with dates and measurement values 
        (format: rows = timesteps, columns = piezometers)
    - sensor_coords: DataFrame containing the coordinates of piezometers.
    
    Returns:
    - A list of numpy arrays, each containing the spatial coordinates and groundwater measurements
      for each timestep.
    """
    z_data_list = [] #list of tacked arrays containing the spatial coordinates and corresponding data values for each timestep
    for i in range(len(spatial_data)): # Iterate through each timestep
        z_array = spatial_data.iloc[i, :].to_numpy().reshape(-1, 1)
        data_coords = sensor_coords.iloc[:, :2].to_numpy()

        valid_indices = ~np.isnan(z_array).squeeze() # Remove locations with NaN values
        valid_z_array= z_array[valid_indices]

        # Make sure both array shapes are compatible
        valid_data_coords= data_coords[valid_indices]

        # Stack the data and the coordinates
        gw_data_stack = np.hstack((valid_data_coords, valid_z_array))
        z_data_list.append(gw_data_stack)

    return z_data_list

#--------------------------------------------------------------------------------------------------------------------------------------
#%% IDW function for 2D arrays with retention of original values
# Calculate distance
def calculate_distances(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points.
    """
    distances = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distances
#--------------------------------------------------------------------------------------------------------------------------------------

# %%
# IDW function for 2D arrays with retention of original values
def idw_2d(source_x:np.array, source_y:np.array, z_values:np.array,
           target_x:np.array, target_y:np.array, power=1):
    """
    Interpolates groundwater values using the Inverse Distance Weighting (IDW) method.
    The function takes in the coordinates of the source and target points and the data 
    to interpolate in a numpy array format.
    Returns the interpolated values in a 2D array.
    source_x: x-coordinates of the source points
    source_y: y-coordinates of the source points
    z_values: values to be interpolated
    source_x, source_y, and z_values must have the same length and are obtained from the .create_interpolation_data function

    target_x: x-coordinates of the target points (gridx)
    target_y: y-coordinates of the target points (gridy)
    power: power parameter for the IDW method. Default is 1.
    Refer to:
    https://pareekshithkatti.medium.com/inverse-distance-weighting-interpolation-in-python-68351fb612d2
    This function takes within it the calculate_distances function.
    """
    # Create an empty array to store the interpolated values
    #Returns an array with same shape as target_x. Use NaN as initial values to easily identify unchanged cells
    interpolated_grid = np.full_like(target_x, np.nan) 

    # Loop through each target cell
    for i in range(target_x.shape[0]):  # Iterate over rows
        for j in range(target_x.shape[1]):  # Iterate over columns
            # Calculate the distance between the target cell and all source cells
            distances = calculate_distances(source_x, source_y, target_x[i, j], target_y[i, j])

            # Check if the target point coincides with a source point
            if np.any(distances == 0):
                # Assign the source point's value to the target point
                interpolated_grid[i, j] = z_values[np.argmin(distances)]
            else:
                # Calculate the weights based on the inverse distance squared
                weights = np.sum(1.0 / (distances**power))

                ## Calculate the weighted sum of known values based on distances:
                numerator = np.sum(z_values/(distances**power))

                # Calculate the interpolated value
                interpolated_grid[i, j] = np.round((numerator/weights),4)

    return interpolated_grid

#--------------------------------------------------------------------------------------------------------------------------------------
# %%


