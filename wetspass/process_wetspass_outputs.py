#multipurpose library
#%%
import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import glob

#%%
# Define the directory where the ASCII files are located
ascii_dir = r"W:/VUB/main_research/model_sims/wetspass/outputs"

wb_variables = ['Cell_evapotranspiration', 'recharge', 'Cell_runoff', 'gwdepth', 'Interception']
#%% Define coordinate arrays of the study area (the order is gridx: minx, min_y, resolution)
#make sure the resolution matches that of your input data (check from the headerfile of the ascii files)
gridx = np.arange(159600, 163350, 10, dtype=np.float32) 
gridy = np.arange(207200, 205500, -10, dtype=np.float32)

# Define the start date of the time series
start_date = "2003-09-30"
time_frequency = "M"

variables_data_arrays = []

# Define a function to extract the numeric part from file names
def extract_number(file_path):
    return int(os.path.splitext(os.path.basename(file_path))[0].split(wb_variable)[1])

# Iterate over each water balance variable
for wb_variable in wb_variables:
    variable_arrays = []
    ascii_files = glob.glob(os.path.join(ascii_dir, f"{wb_variable}*.asc"))

    # Sort the files based on the numeric part of the file names
    sorted_file_paths = sorted(ascii_files, key=extract_number)

    for file in sorted_file_paths:
        # Read the ASCII file as a numpy array
        data = np.loadtxt(file, skiprows=6)

        # Create an xarray DataArray from the numpy array with coordinates and dimensions
        data_array = xr.DataArray(data, coords={"lat": gridy, "lon": gridx}, dims=["lat", "lon"])

        # Set the name of the DataArray to match the water balance variable
        data_array.name = wb_variable
        data_array[wb_variable] = "mm/month"
        variable_arrays.append(data_array)
    
    # Create a time range for the DataArrays
    time_range = pd.date_range(start=start_date, periods=len(ascii_files), freq=time_frequency)

    # Add the time dimension to each DataArray and assign time coordinates
    for i in range(len(variable_arrays)):
        variable_arrays[i] = variable_arrays[i].assign_coords(time=time_range[i])

    # Concatenate the DataArrays along the time dimension
    variables_data_arrays.append(xr.concat(variable_arrays, dim="time"))

# %%
# Convert the variables_data_arrays to a dataset
wb_dataset = xr.Dataset({wb_variables[i]: variables_data_arrays[i] for i in range(len(wb_variables))})
#replace -9999 with NaN
wb_dataset = wb_dataset.where(wb_dataset!=-9999)


# Add metadata to the dataset
wb_dataset.attrs["description"] = "WetsPass model outputs for the Molenbeek catchment."
wb_dataset.attrs["source"] = "WetsPass-M model simulations"
wb_dataset.attrs["units"] = "Except for the groundwater depth (metres), all variables are in mm/month"
wb_dataset.attrs["crs"] = "EPSG:31370"
wb_dataset.attrs["contact"] = "klekarkar@gmail.com"

# Save the dataset to a netCDF file
output_file = os.path.join(ascii_dir, "wetspass_outputs.nc")
#wb_dataset.to_netcdf(output_file)
# %%
vars=['Cell_evapotranspiration', 'recharge', 'Cell_runoff', 'Interception']
var='Cell_runoff'
result = wb_dataset[var].sel(lon=162003.87, lat=205395.85, method='nearest')
df_result=result.to_dataframe()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_result[var], label=var)
#secondary axis
ax2 = ax.twinx()
ax2.invert_yaxis() #for groundwater depth
df_gwdepth=wb_dataset['gwdepth'].sel(lon=162003.87, lat=205395.85, method='nearest').to_dataframe()
ax2.plot(df_gwdepth['gwdepth'], label='gwdepth', color='red')
# plt.axhline(y=0, color='r', linestyle='--')
# plt.legend()

# %%
# Access a specific variable from the dataset
variable_names = list(wb_dataset.variables.keys())

# %%
ax2.plot(df_gwdepth['gwdepth'], label='gwdepth', color='red')
# %%
