# Imports:
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from pymsis import msis
import pandas as pd

# Read in aspect information:
# Note: the cosi aspect file is too large for basic github.
# Contact me if you need help accessing this file. 
aspect_file = "/zfs/astrohe/ckarwin/COSI/Atmoshpere/Run_8/aspect_dso.csv"
df = pd.read_csv(aspect_file)
dates = df["unixtime_full"]
lats = df["latitude"]
lons = df["longitude"]
alts = df["altitude"] / 1000.0 # km

# Convert to numpy arrays:
dates_days_times = np.array(dates, dtype="datetime64") # every second
dates_hrs = np.array(dates, dtype="datetime64[h]") # hrs only 
dates_days = np.array(dates, dtype="datetime64[D]") # days only
lats = np.array(lats).astype(float)
lons = np.array(lons).astype(float)
alts = np.array(alts).astype(float)

# Get rid of bad data points (i.e. entries with all zero):
zero_index = alts == 0
dates_days_times = dates_days_times[~zero_index]
dates_hrs = dates_hrs[~zero_index]
dates_days = dates_days[~zero_index]
lats = lats[~zero_index]
lons = lons[~zero_index]
alts = alts[~zero_index]

# Get mean altitude for entire flight:
alt_mean = np.mean(alts)
alt_std = np.std(alts)
print()
print("mean altitude [km]: " + str(alt_mean) + " +/- " + str(alt_std))

# Average over some time period.
# Here we'll use hours:
lats_mean = []
lons_mean = []
alts_mean = []
unique = np.unique(dates_hrs)
for each in unique:
    this_index = dates_hrs == each
    lats_mean.append(np.mean(lats[this_index]))
    lons_mean.append(np.mean(lons[this_index]))
    alts_mean.append(np.mean(alts[this_index]))
lats_mean = np.array(lats_mean)
lons_mean = np.array(lons_mean)
alts_mean = np.array(alts_mean)

# Get atmosphere model:
output = msis.run(unique,lons_mean,lats_mean,alts_mean)
output = np.squeeze(output)
np.save("cosi_atm",output)
print("atmosphere model shape:")
print(output.shape)

# Plot altitudes:
fig = plt.figure(figsize=(12,4))
ax = plt.gca()
plt.plot(unique,alts_mean,color="navy",label="hr mean")
plt.axhline(y=alt_mean,ls="--",color="orange",label="total mean")
plt.xlabel("Time")
plt.ylabel("Altitude [km]")
plt.grid(ls="--",color="grey",alpha=0.3)
plt.ylim(20,36)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=168))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%D"))
plt.legend(loc=1,ncol=2,frameon=False)
plt.savefig("alt_time_profile_hr_average.pdf")
plt.show()
plt.close()

# Plot latitudes:
fig = plt.figure(figsize=(12,4)) 
ax = plt.gca()
plt.plot(unique,lats_mean,color="navy",label="hr mean")
plt.xlabel("Time")
plt.ylabel("Geographic Latitude [deg]")
plt.grid(ls="--",color="grey",alpha=0.3)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=168))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%D"))
plt.legend(loc=1,ncol=2,frameon=False)
plt.savefig("lat_time_profile_hr_average.pdf")
plt.show()
plt.close()

# Plot mass density:
density_mean = np.mean(output[:,0])
density_std = np.std(output[:,0])
print()
print("Density mean [kg/m3]: " + str(density_mean) + " +/- " + str(density_std))
fig = plt.figure(figsize=(12,4))
ax = plt.gca()
plt.semilogy(unique,output[:,0],color="navy",label="hr mean")
plt.axhline(y=density_mean,ls="--",color="orange",label="total mean")
plt.xlabel("Time")
plt.ylabel("Mass Density [$\mathrm{kg \ m^{-3}}$]")
plt.grid(ls="--",color="grey",alpha=0.3)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=168))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%D"))
plt.legend(loc=1,ncol=2,frameon=False)
plt.savefig("density_time_profile_hr_average.pdf")
plt.show()
plt.close()
