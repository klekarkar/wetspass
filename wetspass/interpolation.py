#%%import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import pykrige.kriging_tools as kt


#%%import and preprocess data
gw_data=pd.read_excel('../data/molenbeek_bgk/piezometer_levels.xlsx', sheet_name='DATA')
elevation_data=pd.read_excel('../data/molenbeek_bgk/GWTable.xlsx', sheet_name='GWs_Elev_data')
sensor_coords=pd.read_csv("../data/molenbeek_bgk/peil_locations_c.csv")
gw_data_refined=gw_data.drop('COMMENTAAR', axis=1)

# convert period column to datetime
gw_data_refined['date']=pd.to_datetime(gw_data_refined['periode (halve maand)'], format='%Y_%m_%d')
gw_data_refined=gw_data_refined.drop('periode (halve maand)',axis=1)


#%%sort the data by date
gw_data_refined=gw_data_refined.sort_values('date').set_index('date')
gw_monthly_depth=gw_data_refined.resample('M').mean()*0.01*-1 # resample to monthly and convert from cm to meters. -1 converts measurements to depth below surface


#%%
"""the data is measured twice a month on the first and second half. so we assume the first monthly 
measurement is taken on the first day and the 2nd measurement on the 15th day"""

mask = gw_data_refined.index.day == 2

if mask.any():
    gw_data_refined.index = gw_data_refined.index.where(~mask, gw_data_refined.index + pd.DateOffset(days=13))

elevation_data=elevation_data.loc[:,['NAME','ELEV_1']]


#%%KRIGGING
#import a DEM of the area to be used as interpolation mask. resample to 10m resolution
dem_data=np.loadtxt("../data/molenbeek_bgk/R5.asc", skiprows=6)

#mask array. I assume there are no negative elevations in the DEM, so i mask all 'real elevation' values with 1 and the rest remains as is.
data_mask=np.where(dem_data>=0, 1, dem_data)

#%%
#get groundwater data for each month and convert to np array and transpose to single column
#iloc selects all the rows and the first column (the first month of data). This is the value that will need to be modified when looping through each timestep
#the df is has piezometers as rows and the monthly data as columns.

for i in range(len(gw_monthly_depth)): #iterate through each timestep
    gw_monthly_array=gw_monthly_depth.iloc[i,:].to_numpy().reshape(-1,1) #reshape converts the array to a single column so that each row can associated with a coordinate

    '''sensor coordinates. The first two columns are the x and y coordinates. The rest are not needed, hence left out
        sensor coords is a dataframe containing the coordinates of piezometers'''
    data_coords=sensor_coords.iloc[:,:2].to_numpy() 

    # The data: x,y,z, where z (gw data) is the value to be interpolated.
    #get rid of the nan values
    valid_indices = ~np.isnan(gw_monthly_array) #remove locations with nan values. it causes an error in the interpolation
    gw_monthly_array_clean = gw_monthly_array[valid_indices]

    #extract the coordinates of the valid indices. piezometer with nan values are removed
    #make sure both array shapes are compatible
    #
    data_coords_clean = data_coords.reshape(-1,2)[valid_indices[:, 0]] #valid_indices is a 2D boolean array. Select the first column to filter the coordinates to get rid of nan measurements.

    #stack the data and the coordinates
    gw_data_stack = np.hstack((data_coords_clean, gw_monthly_array_clean.reshape(-1, 1)))


    # Generate grid points based on the size of the mask and the resolution of the DEM
    #Gridpoints correspond to the bounds of the DEM/study area to interpolate the groundwater data (BE Lambert Coordinate system). 5 is the resolution of the DEM.
    gridx = np.arange(159700, 163295, 5,dtype=np.float64) #krigging performs operations on un
    gridy = np.arange(205520,207160, 5,dtype=np.float64)

    # Perform Ordinary Kriging. I grabbed the code from the pykrige documentation
    '''https://geostat-framework.readthedocs.io/projects/pykrige/en/latest/examples/00_ordinary.html'''
    # Create the ordinary kriging object. Required inputs are the X-coordinates of the data points, the Y-coordinates of the data points, and the Z-values of the data points.
    OK = OrdinaryKriging(
        gw_data_stack[:, 0],  # x-values
        gw_data_stack[:, 1],  # y-values
        gw_data_stack[:, 2],  # z-values (groundwater data)
        variogram_model="linear",  # Adjust variogram model as needed
        verbose=False,
        enable_plotting=False,
    )

    # Execute kriging to generate the interpolated values on the grid
    z, ss = OK.execute("grid", gridx, gridy)

    #write the interpolated data to the DEM mask, in my case 1 represents all grids inside the study area. use -9999 as no data value
    interpolated_gw=np.where(data_mask==1, z, -9999)
    interpolation_error=np.where(data_mask==1, ss, -9999)

    #write data to ascii grid
    data_catchment_=np.flip(interpolated_gw, axis=0) #somehow the output is flipped about the y-axis. This is a quick fix
    kt.write_asc_grid(gridx, gridy, data_catchment_, filename=f"interpolated_gw_ts_{i}.asc", style=2)


#%% Visualize interpolated values
interpolated_gw_values=np.where(interpolated_gw>=0, interpolated_gw, np.nan)
plt.imshow(interpolated_gw_values, extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()))
plt.plot(gw_data_stack[:, 0], gw_data_stack[:, 1], "g.", markersize=5) #coorinates
plt.grid(alpha=0.4, c='gray', linewidth=0.2)
plt.colorbar(label="groundwater depth(m)", orientation="horizontal")
plt.xlabel("Lat (m)")
plt.ylabel("Lon (m)")
plt.title("Interpolated Groundwater Levels")
plt.show()