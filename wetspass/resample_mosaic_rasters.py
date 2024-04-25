#%% 
"""
code to do multiple raster operations
"""
#use the spatialenv conda environment
import rasterio
import os
import glob
from osgeo import gdal
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
from scipy.ndimage import distance_transform_edt

# %%
def resample_raster_image(src_directory, downscale_factor, output_directory, no_data=-9999):
    """Resamples all TIFF images in the source directory by a given downscale factor, properly handling no data values."""
    src_files_to_resample = [f for f in os.listdir(src_directory) if f.endswith(".tif")]

    for dem in src_files_to_resample:
        src_path = os.path.join(src_directory, dem)
        if os.path.isfile(src_path):
            with rasterio.open(src_path) as dataset:
                # Set no data value for the dataset if not already set
                if dataset.nodata is None:
                    dataset.nodata = no_data

                new_height = int(dataset.height * downscale_factor)
                new_width = int(dataset.width * downscale_factor)

                # Ensure the resampling process considers the no data value
                data = dataset.read(
                    out_shape=(dataset.count, new_height, new_width),
                    resampling=Resampling.bilinear,
                    masked=True  # Use a masked array to ignore no data values
                )

                # Calculate the new transform
                transform = dataset.transform * dataset.transform.scale(
                    (dataset.width / data.shape[-1]),
                    (dataset.height / data.shape[-2])
                )

                out_meta = dataset.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": new_height,
                    "width": new_width,
                    "transform": transform,
                    "nodata": no_data  # Ensure the output file also has the no data value set
                })

                output_path = os.path.join(output_directory, f"downscaled_{dem}")
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(data)

# Example usage:
#resample_raster_image(par_dir, downscale_factor, output_dir)
#%% code to mosaic rasters
src_files_to_mosaic=[]
for tif_file in os.listdir(output_dir):
    if tif_file.endswith(".tif"):
        src_files_to_mosaic.append(rasterio.open(os.path.join(output_dir,tif_file)))    #open and append files to the mosaic list

#%% mosaic the files
mosaic, out_trans = merge(src_files_to_mosaic)

    
# %%
show(mosaic, cmap='terrain')

# %%
# Copy the metadata
out_meta = rasterio.open(tif_file).meta.copy()

# Update the metadata
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans,
    "crs": "EPSG:31370"
})

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%
def convert_tif_raster_to_ascii(tif_path, out_ascii_path):
    """Convert a GeoTIFF raster to ASCII format."""
    try:
        subprocess.run(['gdal_translate', '-of', 'AAIGrid', '-ot', 'Float32',
                        '-co', 'DECIMAL_PRECISION=3', '-a_nodata', '-9999',
                        tif_path, out_ascii_path], check=True)
        print("Conversion successful.")
    except subprocess.CalledProcessError as e:
        print("Error during conversion:", e)
        sys.exit(1)
    except Exception as e:
        print("An unexpected error occurred:", e)
        sys.exit(1)


#*******************************************************************************************

# %%reclassify raster
def reclassify_raster(input_raster_path, in_raster, reclassification_dict):
    """Reclassify a raster based on a reclassification dictionary."""

    # Open the input raster
    with rasterio.open(os.path.join(input_raster_path,in_raster)) as dataset:
        # Read the data from the input
        data = dataset.read(1) #1 represents the first band
        
        #numpy vectorize function to reclassify the data
        vectorize_map=np.vectorize(reclassification_dict.get)

        reclassified_raster = vectorize_map(data)
        output_raster=os.path.join(input_raster_path,'reclassified.tif')


        # Write the new data to a new raster file
        # Open the original raster to copy the metadata
    with rasterio.open(os.path.join(input_raster_path,in_raster)) as src:
        meta = src.meta

    # Update the metadata (if necessary)
    meta.update(count=1)

    # Write the reclassified raster
    with rasterio.open(output_raster, 'w', **meta) as dst:
        dst.write(reclassified_raster,1)

# %%
par_dir=r'W:\VUB\main_research\data\molenbeek_bgk\landuse'
in_raster='beggelbeek_lu_10m.asc'
reclassification_dict = {
    1: 1, 2: 3,3: 3,4: 1,
    5: 201,6: 2,7: 3,8: 2,9:307,10:7,
    11: 6,12: 31,13: 21,14: 21,
    15: 31,16: 23,17: 52,18: 44,
    19: 28,-9999: -9999}


#%% fill na with nearest neighbor
def fill_na_nearestneighbor(raster_path):

    # Load raster
    raster = gdal.Open(raster_path)
    raster_array = raster.ReadAsArray()

    # Create a mask of NaN values in the raster
    mask = (raster_array == -9999)

    # Compute the distance transform of the mask
    dist_transform = distance_transform_edt(mask)

    # Replace NaN values with the nearest neighbor value
    filled_raster = np.where(mask, raster_array[np.unravel_index(dist_transform.argmin(), raster_array.shape)], raster_array)

    return filled_raster

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Load raster
raster = gdal.Open(r"W:\VUB\main_research\data\molenbeek_bgk\soiltype_lb72\soil_with_nan.tif")
raster_array = raster.ReadAsArray()

# Create a mask of NaN values in the raster
mask = (raster_array == -9999)

# Compute the distance transform of the mask
dist_transform = distance_transform_edt(mask)

# Replace NaN values with the nearest neighbor value
filled_raster = np.where(mask, raster_array[np.unravel_index(dist_transform.argmin(), raster_array.shape)], raster_array)






# %%
reclassify_raster(par_dir, in_raster, reclassification_dict)

# %%
src_dir=r"W:\VUB\main_research\data\molenbeek_bgk\dem"
output_directory=r"W:\VUB\main_research\data\molenbeek_bgk\dem"
#resample factor of 2 means 2x the number of pixels i.e higher resolution

resample_raster_image(src_dir, 0.5, output_directory, no_data=-9999)
# %%
in_=r"W:\VUB\main_research\data\molenbeek_bgk\dem\beggelbeek_10m.tif"
out=r"W:\VUB\main_research\data\molenbeek_bgk\dem\beggelbeek_10m_.asc"
nodata=-9999

convert_tif_raster_to_ascii(in_, out)
# %%

