import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import subprocess
from typing import List
from spatial_interpolation import create_interpolation_data,calculate_distances,idw_2d


"""path to source directory"""
src=r"W:/VUB/main_research/data/molenbeek_bgk"

"""import and preprocess data"""

gw_data=pd.read_excel(os.path.join(src,"peilbuizen_2024_corrected.xlsx"), sheet_name="DATA")
elevation_data=pd.read_excel(os.path.join(src,"GWTable.xlsx"), sheet_name="GWs_Elev_data")
sensor_coords=pd.read_csv(r"W:/VUB/main_research/data/molenbeek_bgk/peil_locations_c.csv")


gw_data_int=gw_data.drop('COMMENTAAR', axis=1)

# convert period column to datetime
gw_data_int['date']=pd.to_datetime(gw_data_int['periode (halve maand)'], format='%Y_%m_%d')

#sort the data by date
gw_df_processed=gw_data_int.sort_values('date').set_index('date').drop('periode (halve maand)',axis=1)



"""the data is measured twice a month on the first and second half. so we assume the first monthly 
measurement is taken on the first day and the 2nd measurement on the 15th day"""

mask = gw_df_processed.index.day == 2

if mask.any():
    gw_df_processed.index = gw_df_processed.index.where(~mask, gw_df_processed.index + pd.DateOffset(days=13))

gw_monthly_depth=gw_df_processed.resample('M').mean()*0.01*-1 # resample to monthly and convert from cm to meters. "*-1" converts measurements to depth below surface
gw_monthly_depth.head()


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""interpolate the data"""

#generate a grid of coordinates to be used for interpolation
#Gridpoints correspond to the bounds of the DEM/study area to interpolate the groundwater data (BE Lambert Coordinate system). 5 is the resolution of the interpolation space
gridx = np.arange(159600, 163350, 10,dtype=np.float64) 
gridy = np.arange(207200,205500,-10, dtype=np.float64)

target_grid=np.meshgrid(gridx, gridy)


#create interpolation data from the create interpolation data function
#the function takes the groundwater data and the sensor coordinates as input and returns a list of arrays containing the sensor coordinates and the groundwater data. Eac array corresponds to a timestep
gw_data_stack_list=create_interpolation_data(gw_monthly_depth,sensor_coords)

#iterate through each timestep(array) and interpolate the groundwater data
gw_gridded_list= []
for i, gw_data_stack in enumerate(gw_data_stack_list):
    source_x, source_y, gw_values = gw_data_stack[:, 0], gw_data_stack[:, 1], gw_data_stack[:, 2]
    target_x, target_y = np.meshgrid(gridx, gridy)

    # Interpolate the groundwater data using IDW
    interpolated_array = idw_2d(source_x, source_y, gw_values, target_x, target_y, power=2)

    # Convert the NumPy array to an xarray DataArray
    xarray_gw = xr.DataArray(interpolated_array, coords=[gridy, gridx], dims=["lat", "lon"])

    # Convert the DataArray to a Dataset
    gw_dataset = xr.Dataset({"groundwater_depth": xarray_gw})

    gw_gridded_list.append(gw_dataset)

    # Update progress bar
    # Update progress bar
    progress = (i + 1) / len(gw_data_stack_list)
    bar_length = 20  # Adjust this for longer or shorter bar
    bar = "#" * int(bar_length * progress)
    print(f"\rProcessing: [{bar.ljust(bar_length)}] {int(progress * 100)}%", end="")

print("\nInterpolation completed.")

"""
Write the data to an xarray dataset
"""

# Define the start and end dates
start_date = "2003-09-30"

# Create a time range from start to end with monthly frequency
time_range = pd.date_range(start=start_date, periods=len(gw_data_stack_list), freq="M")

# Assign the time range to the gw_dataset
for i in range(len(time_range)):
    gw_gridded_list[i]["time"] = time_range[i]

#concatenate the datasets along the time dimension
gw_gridded_dataset=xr.concat(gw_gridded_list, dim="time")
#assign global attributes to the dataset and specific attributes to the DataArray
gw_gridded_dataset.attrs["crs"] = "EPSG:31370"
gw_gridded_dataset["groundwater_depth"].attrs["units"] = "metres"
gw_gridded_dataset.attrs['description'] = 'Groundwater depths interpolated spatially using IDW method at 10m resolution.'
gw_gridded_dataset.attrs['source'] = 'Groundwater data from the Beggelbeek groundwater monitoring wells.'
gw_gridded_dataset.attrs['author'] = 'Katoria Lekarkar'


"""
Export the files to ascii grids
"""
#GitHub Copilot
# Convert the NetCDF file to an ASCII grid file using gdal_translate
# Construct the file paths
dest_folder=r'W:/VUB/main_research/data/molenbeek_bgk/gridded_gw_depth'
if not os.path.exists(os.path.join(os.getcwd(),dest_folder)):
    os.makedirs(dest_folder)

for i in range(len(gw_gridded_dataset['time'])):
    # Construct the file paths
    netcdf_file_path = os.path.join(dest_folder, f"gwdepth{i+1}.nc")
    ascii_file_path = os.path.join(dest_folder, f"gwdepth{i+1}.asc")

    gw_gridded_dataset.isel(time=i).to_netcdf(netcdf_file_path)
    # Convert the NetCDF file to an ASCII grid file using gdal_translate
    subprocess.run(['gdal_translate', '-of', 'AAIGrid', '-ot', 'Float32', '-co', 'DECIMAL_PRECISION=4','-a_nodata', '-9999', netcdf_file_path, ascii_file_path])

    # Remove the temporary NetCDF file
    os.remove(netcdf_file_path)


""""
Perform a quick accuracy check of the interpolated data
"""
#chose one of the piezometers and plot the interpolated and measured groundwater depth
fig, ax = plt.subplots(figsize=(10, 3))
data=gw_gridded_dataset['groundwater_depth'].sel(lon=162003.87, lat=206395.85, method='nearest')
df=pd.DataFrame(data.values)
df['time']=gw_gridded_dataset['time']
df.set_index('time', inplace=True)
plt.plot(df.index, df.values)
plt.plot(gw_monthly_depth['C3']) #convert to meters and flip the sign
ax.invert_yaxis()
plt.legend(['Interpolated','Measured'])
plt.ylabel('groundwater depth (m)')