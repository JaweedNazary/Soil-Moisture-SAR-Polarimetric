import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np

#================================================================#
#          Baghdadi et al 2016 semi empirical model              #
#================================================================#
def baghdadi_2016(ks : float,pq : str, theta_deg : float, mv_pct : float):
    """
    Computes radar backscatter based on a semi empirical model for bare soil from Baghdadi et al., 2016
    "A New Empirical Model for Radar Scattering from Bare Soil Surfaces, Baghdadi et al., 2016"
    text avaible here: https://www.mdpi.com/2072-4292/8/11/920/htm
    """
    theta_rad = theta_deg * np.pi / 180
    # Baghdadi model parameters for hh, vv and hv polarizations  
    if pq == 'hh':
        delta,beta,gamma,xi=-1.287,1.227,0.009,0.86 
    elif pq == 'vv':
        delta,beta,gamma,xi=-1.138,1.528,0.008,0.71 
    elif pq == 'hv':
        delta,beta,gamma,xi=-2.325,0.01,0.011,0.44
    else:
        print("polarization %s not supported" %pq)

    
    sigma    = (10 ** delta) * ((np.cos(theta_rad)) ** beta) * (10 ** (gamma * (1/np.tan(theta_rad)) * mv_pct)) \
                   * (ks ** (xi * np.sin(theta_rad)))
    sigma_dB =  10 * np.log10(sigma)
    return sigma_dB


#================================================================#
#          Validation of Baghdadi et al 2016 model               #
#================================================================#

import matplotlib.pyplot as plt
qp = "hh"
tetha = 25
x = np.arange(0.1, 6, 0.001)
y = baghdadi_2016(x,qp,tetha,5)
y1 = baghdadi_2016(x,qp,tetha,15)
y2 = baghdadi_2016(x,qp,tetha,35)

fig, ax = plt.subplots(3,2, figsize=(12,13))
ax[0][0].plot(x, y, label="mv = 5%")
ax[0][0].plot(x, y1, label="mv = 15%")
ax[0][0].plot(x, y2, label="mv = 35%")
ax[0][0].set(xlabel='k Hrms (cm)', ylabel='backscattering %s (dB)' %qp)
ax[0][0].text(3,-20,"Incident angle = $%s^o$" %str(tetha))
ax[0][0].grid()
ax[0][0].set_ylim(-25, 0)
ax[0][0].set_xlim(0, 6)
ax[0][0].legend()

qp = "hh"
tetha = 45
x = np.arange(0.1, 6, 0.001)
y = baghdadi_2016(x,qp,tetha,5)
y1 = baghdadi_2016(x,qp,tetha,15)
y2 = baghdadi_2016(x,qp,tetha,35)

ax[0][1].plot(x, y, label="mv = 5%")
ax[0][1].plot(x, y1, label="mv = 15%")
ax[0][1].plot(x, y2, label="mv = 35%")
ax[0][1].set(xlabel='k Hrms (cm)', ylabel='backscattering %s (dB)' %qp)
ax[0][1].text(3,-20,"Incident angle = $%s^o$" %str(tetha))
ax[0][1].grid()
ax[0][1].set_ylim(-25, 0)
ax[0][1].set_xlim(0, 6)
ax[0][1].legend()

qp = "vv"
tetha = 25
x = np.arange(0.1, 6, 0.001)
y = baghdadi_2016(x,qp,tetha,5)
y1 = baghdadi_2016(x,qp,tetha,15)
y2 = baghdadi_2016(x,qp,tetha,35)

ax[1][0].plot(x, y, label="mv = 5%")
ax[1][0].plot(x, y1, label="mv = 15%")
ax[1][0].plot(x, y2, label="mv = 35%")
ax[1][0].set(xlabel='k Hrms (cm)', ylabel='backscattering %s (dB)' %qp)
ax[1][0].text(3,-20,"Incident angle = $%s^o$" %str(tetha))
ax[1][0].grid()
ax[1][0].set_ylim(-25, 0)
ax[1][0].set_xlim(0, 6)
ax[1][0].legend()

qp = "vv"
tetha = 45
x = np.arange(0.1, 6, 0.001)
y = baghdadi_2016(x,qp,tetha,5)
y1 = baghdadi_2016(x,qp,tetha,15)
y2 = baghdadi_2016(x,qp,tetha,35)

ax[1][1].plot(x, y, label="mv = 5%")
ax[1][1].plot(x, y1, label="mv = 15%")
ax[1][1].plot(x, y2, label="mv = 35%")
ax[1][1].set(xlabel='k Hrms (cm)', ylabel='backscattering %s (dB)' %qp)
ax[1][1].text(3,-20,"Incident angle = $%s^o$" %str(tetha))
ax[1][1].grid()
ax[1][1].set_ylim(-25, 0)
ax[1][1].set_xlim(0, 6)
ax[1][1].legend()

qp = "hv"
tetha = 25
x = np.arange(0.1, 6, 0.001)
y = baghdadi_2016(x,qp,tetha,5)
y1 = baghdadi_2016(x,qp,tetha,15)
y2 = baghdadi_2016(x,qp,tetha,35)

ax[2][0].plot(x, y, label="mv = 5%")
ax[2][0].plot(x, y1, label="mv = 15%")
ax[2][0].plot(x, y2, label="mv = 35%")
ax[2][0].set(xlabel='k Hrms (cm)', ylabel='backscattering %s (dB)' %qp)
ax[2][0].text(3,-30,"Incident angle = $%s^o$" %str(tetha))
ax[2][0].grid()
ax[2][0].set_ylim(-35, -10)
ax[2][0].set_xlim(0, 6)
ax[2][0].legend()

qp = "hv"
tetha = 45
x = np.arange(0.1, 6, 0.001)
y = baghdadi_2016(x,qp,tetha,5)
y1 = baghdadi_2016(x,qp,tetha,15)
y2 = baghdadi_2016(x,qp,tetha,35)

ax[2][1].plot(x, y, label="mv = 5%")
ax[2][1].plot(x, y1, label="mv = 15%")
ax[2][1].plot(x, y2, label="mv = 35%")
ax[2][1].set(xlabel='k Hrms (cm)', ylabel='backscattering %s (dB)' %qp)
ax[2][1].text(3,-30,"Incident angle = $%s^o$" %str(tetha))
ax[2][1].grid()
ax[2][1].set_ylim(-35, -10)
ax[2][1].set_xlim(0, 6)
ax[2][1].legend()

plt.show()



#================================================================#
#          Solving Baghdadi et al 2016 model for moisture %      #
#================================================================#

def Baghdadi_reverse_2016(pq1_dB, pq2_dB, pq1_type, pq2_type ,theta_deg):
    """
    Computes mositure content based on the reversed semi empirical model for bare soil from Baghdadi et al., 2016
    "A New Empirical Model for Radar Scattering from Bare Soil Surfaces, Baghdadi et al., 2016"
    text avaible here: https://www.mdpi.com/2072-4292/8/11/920/htm
    
    All right reserved to M. Jaweed Nazary University of Missouri 10/14/2022 
    
    pq1 = first radar backscatter coefficient in dB units. Must be one of (HH, VV, and HV polarization modes)
    pq2 = second radar backscatter coefficient in dB units. Must be one of (HH, VV, and HV polarization modes)
    pq1_type = str, from list ["hh", "vv", "hv"]
    pq2_type = str, from list ["hh", "vv", "hv"]
    theta_deg = incidence angle in degree

    email: jaweedpy@gmail.com
    
    """
    theta_rad = theta_deg * np.pi / 180.0
    
    # converting dB to linear scale 
    pq1 = np.power(10,(pq1_dB/10))
    pq2 = np.power(10,(pq2_dB/10))
    
    # Calculating the parameters for HH 
    f_hh = (np.power(10, -1.287)) * (np.power(np.cos(theta_rad), 1.227))
    B_hh = 0.009 * (1/np.tan(theta_rad))
    C_hh = 0.86 * np.sin(theta_rad)
    e_hh = np.divide(B_hh,C_hh)
    
    # Calculating the parameters for VV 
    f_vv = (np.power(10 , -1.138)) * (np.power(np.cos(theta_rad), 1.528))
    B_vv = 0.008 * (1/np.tan(theta_rad))
    C_vv = 0.71 * np.sin(theta_rad)
    A_vv = np.power((pq1/f_vv),(1/C_vv))
    e_vv = np.divide(B_vv,C_vv)
    
    # Calculating the parameters for HV 
    f_hv = (np.power(10 , -2.325)) * (np.power(np.cos(theta_rad), 0.01))
    B_hv = 0.011 * (1/np.tan(theta_rad))
    C_hv = 0.44 * np.sin(theta_rad)
    A_hv = np.power((pq1/f_hv),(1/C_hv))
    e_hv = np.divide(B_hv,C_hv)
    
    # checking if the first data layer is hh, vv, or hv
    if pq1_type == "hh":
        A_hh = np.power((pq1/f_hh),(1/C_hh))
        A1 = A_hh
        e1 = e_hh
    elif pq1_type == "vv":
        A_vv = np.power((pq1/f_vv),(1/C_vv))
        A1 = A_vv
        e1 = e_vv
    elif pq1_type == "hv":
        A_hv = np.power((pq1/f_hv),(1/C_hv))
        A1 = A_hv
        e1 = e_hv        
    
    # checking if the second data layer is hh, vv, or hv
    if pq2_type == "hh":
        A_hh = np.power((pq2/f_hh),(1/C_hh))
        A2 = A_hh
        e2 = e_hh
    elif pq2_type == "vv":
        A_vv = np.power((pq2/f_vv),(1/C_vv))
        A2 = A_vv
        e2 = e_vv
    elif pq2_type == "hv":
        A_hv = np.power((pq2/f_hv),(1/C_hv))
        A2 = A_hv
        e2 = e_hv
    
    # calculating moisture content
    mv = (np.log(A1/A2))/(e1-e2)
    return mv\



#==============================================================================#
#    saving the water mask    #
#==============================================================================#

# This code block calculates the moisture content based on HH, VV and Incidence Angle of SAR Image. 
# the input image need to be calibrated and transformed to dB units of measurement.

from scipy.signal import medfilt
from rasterio import Affine
from rasterio.enums import Resampling

with rasterio.open('ALOS__L1_2009_SAR.tif') as src:
    HH_db = src.read(1)
    tetha = src.read(7)
    
    # threshold using Chopman et al 2015 for detection of indunation 
    threshold = np.add(np.multiply(-0.1266,tetha),-8.2)
    water = np.less_equal(HH_db,threshold)
    water = np.multiply(water,1)
    water = water.reshape(1,HH_db.shape[0],HH_db.shape[1])
    
    # pulling the transformation info from source image
    dst_transform = src.transform
    # defining keyword info for raster creation
    kwargs = {'driver': 'GTiff',
              'dtype': 'float32',
              'nodata': None,
              'width': HH_db.shape[1],
              'height': HH_db.shape[0],
              'count': 1,
              'crs': src.crs,
              'transform': src.transform}

    with rasterio.open('../water.TIF','w', **kwargs) as dst:
        dst.write(water)


#==============================================================================#
#           Reprojecting WATER to EPSG:3857 Pseudo-Mercator projection               #
#==============================================================================#
# this code block reproject the moisture content raster and writes a new raster file

from rasterio.warp import calculate_default_transform, reproject, Resampling

# the destination projected coordinate system. 
# for the porpuse of this project we are using Pseudo-Mercator projection, more info here: 'https://epsg.io/3857'
dst_crs={'init': 'EPSG:3857'}

with rasterio.open('../water.TIF') as src:
    transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    with rasterio.open('../WATER_Projected.TIF', 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

#==============================================================================#
#    saving the result moisutre raster with its transformation information     #
#==============================================================================#

# This code block calculates the moisture content based on HH, VV and Incidence Angle of SAR Image. 
# the input image need to be calibrated and transformed to dB units of measurement.

from scipy.signal import medfilt
from rasterio import Affine
from rasterio.enums import Resampling

with rasterio.open('ALOS__L1_2009_SAR.tif') as src:
    HH_db = src.read(1)
    HV_db = src.read(2) 
    VH_db = src.read(3) 
    VV_db = src.read(4)
    tetha = src.read(7)
    
    # using Baghdadi 2016 model to calculate moisture content
    mv = Baghdadi_reverse_2016(HH_db+0.00001,VV_db+0.00001,"hh","vv",tetha+0.000001)
    mv = medfilt(mv,5)
    mv = mv.reshape(1,HH_db.shape[0],HH_db.shape[1])
    
    # pulling the transformation info from source image
    dst_transform = src.transform
    # defining keyword info for raster creation
    kwargs = {'driver': 'GTiff',
              'dtype': 'float32',
              'nodata': None,
              'width': HH_db.shape[1],
              'height': HH_db.shape[0],
              'count': 1,
              'crs': src.crs,
              'transform': src.transform}

    with rasterio.open('../moisture_content.TIF','w', **kwargs) as dst:
        dst.write(mv)


#==============================================================================#
#           Reprojecting to EPSG:3857 Pseudo-Mercator projection               #
#==============================================================================#
# this code block reproject the moisture content raster and writes a new raster file

from rasterio.warp import calculate_default_transform, reproject, Resampling

# the destination projected coordinate system. 
# for the porpuse of this project we are using Pseudo-Mercator projection, more info here: 'https://epsg.io/3857'
dst_crs={'init': 'EPSG:3857'}

with rasterio.open('../moisture.TIF') as src:
    transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    with rasterio.open('../projected_moisture_content.TIF', 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
with rasterio.open('../projected_moisture_content.TIF') as src:
    mv = src.read(1)
    %matplotlib inline
    fig, ax= plt.subplots(figsize=(45,15))
    plt.imshow(mv,cmap="Blues", vmin=-100,vmax=200)
    colo = plt.colorbar(ax=ax)
    colo.ax.set_ylabel('mv')
