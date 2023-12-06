#%%
import os
import cartopy.crs as ccrs
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ======= Projection for plot  ========
data = nc.Dataset('data/u10_lag5_to_lead9_waccm6_exp1_regrid_bilin.nc','r')
lon = data.variables['lon'][:].flatten()
lat = data.variables['lat'][:].flatten()
map_proj = ccrs.Orthographic(central_longitude=0, central_latitude=90)
data_crs = ccrs.PlateCarree()
x, y, _ = map_proj.transform_points(ccrs.PlateCarree(), lon, lat).T
mask = np.invert(np.logical_or(np.isinf(x), np.isinf(y)))
x = np.compress(mask, x)
y = np.compress(mask, y)

#%%
# ===========================================
# Plot
# ===========================================

# 2D U data
U_SSW = data.variables['u_lag0'][:]  # shape:780*36*36 (EXP2)
U_nSSW = data.variables['u_clim'][:]
Umean_SSW = np.mean(U_SSW, axis=0)   # composite of all EXP2 cases
Umean_nSSW = np.mean(U_nSSW, axis=0)
Umean_SSW = Umean_SSW.flatten()
Umean_nSSW = Umean_nSSW.flatten()
# Directory of classification results
resDir = 'models/validation_result_U/'

# ======= Plot =======
levels = np.linspace(-40., 40., 11)
levels = np.delete(levels, np.where(levels==0)[0][0])
zeros_len = [3,5,7,9,11,13]
title = ['(a)','(b)','(c)','(d)','(e)','(f)']

# Plot: SSW
fig, ax = plt.subplots(ncols=3, nrows=2, subplot_kw={'projection':map_proj},\
                       facecolor='w', figsize=[10,9])
ax = ax.flatten()
for i in range(len(zeros_len)):
    zeros_ang = round(zeros_len[i]*2.07,1)
    errdata = np.load(resDir + 'CNN_interpret_masksize{}.npz'.format(zeros_len[i]))
    error_sum_XY_SSW = errdata['error_sum_XY_SSW']
    num_of_tests = errdata['num_of_tests']
    error_sum_SSWm = error_sum_XY_SSW/num_of_tests*100  # percentage

    im1 = ax[i].tripcolor(x,y, error_sum_SSWm[mask], vmax=20, vmin=0, cmap='Reds', alpha=.6)
    ax[i].coastlines(resolution='110m', color='dimgrey', lw=.8)
    ct = ax[i].tricontour(x,y, Umean_SSW[mask], extend=ax[i].get_extent(),levels=levels,colors='k',linewidths=.75)
    ct0 = ax[i].tricontour(x,y, Umean_SSW[mask], extend=ax[i].get_extent(),levels=[0],colors='k',linewidths=1.5)
    gl = ax[i].gridlines(draw_labels=False, color='mediumblue', lw=1)
    gl.ylocator = mticker.FixedLocator([60])
    gl.xlines = False
    ax[i].clabel(ct, inline=True, fontsize=6)
    ax[i].clabel(ct0, inline=True, fontsize=8)
    ax[i].set_title('mask size = {}x{} (~{}°x{}°)'\
                    .format(zeros_len[i],zeros_len[i],zeros_ang,zeros_ang),fontsize=11)
    ax[i].set_title(title[i],loc='left', fontsize=12,fontweight='bold')

fig.subplots_adjust(bottom=0.35, top=0.95, left=0.2, right=0.8,
        wspace=0.1,hspace=0)
plt.tight_layout()
cax = fig.add_axes([0.15, 0.05, 0.7, 0.025])
cb = plt.colorbar(im1, cax=cax , orientation='horizontal', extend='max')
cb.set_label('% (out of {} tests)'.format(int(num_of_tests)))
plt.tight_layout()
plt.savefig('figures/CNN_SSW_errorplot_U.png', dpi=250)

#%%
# Plot: non-SSW
plt.clf()
fig, ax = plt.subplots(ncols=3, nrows=2, subplot_kw={'projection':map_proj},\
                       facecolor='w', figsize=[10,9])
ax = ax.flatten()
for i in range(len(zeros_len)):
    zeros_ang = round(zeros_len[i]*2.07,1)
    errdata = np.load(resDir + 'CNN_interpret_masksize{}.npz'.format(zeros_len[i]))
    error_sum_XY_nSSW = errdata['error_sum_XY_nSSW']
    num_of_tests = errdata['num_of_tests']
    error_sum_nSSWm = error_sum_XY_nSSW/num_of_tests*100  # percentage

    im1 = ax[i].tripcolor(x,y, error_sum_nSSWm[mask], vmax=30, vmin=0, cmap='Reds', alpha=.6)
    ax[i].coastlines(resolution='110m', color='dimgrey', lw=.8)
    ct = ax[i].tricontour(x,y, Umean_nSSW[mask], extend=ax[i].get_extent(),levels=levels,colors='k',linewidths=.75)
    ct0 = ax[i].tricontour(x,y, Umean_nSSW[mask], extend=ax[i].get_extent(),levels=[0],colors='k',linewidths=1.5)
    gl = ax[i].gridlines(draw_labels=False, color='mediumblue', lw=1)
    gl.ylocator = mticker.FixedLocator([60])
    gl.xlines = False
    ax[i].clabel(ct, inline=True, fontsize=6)
    ax[i].clabel(ct0, inline=True, fontsize=8)
    ax[i].set_title('mask size = {}x{} (~{}°x{}°)'\
                    .format(zeros_len[i],zeros_len[i],zeros_ang,zeros_ang),fontsize=11)
    ax[i].set_title(title[i],loc='left', fontsize=12,fontweight='bold')

fig.subplots_adjust(bottom=0.35, top=0.95, left=0.2, right=0.8,
        wspace=0.1,hspace=0)
plt.tight_layout()
cax = fig.add_axes([0.15, 0.05, 0.7, 0.025])
cb = plt.colorbar(im1, cax=cax , orientation='horizontal', extend='max')
cb.set_label('% (out of {} tests)'.format(int(num_of_tests)))
plt.tight_layout()
plt.savefig('figures/CNN_nSSW_errorplot_U.png', dpi=250)
