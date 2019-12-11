import instaseis
import obspy
import numpy as np
import matplotlib
import iterdecon as rfunc
import seisutils as su
from obspy.taup import TauPyModel
from matplotlib import rc
from scipy import signal
from mpl_toolkits.basemap import Basemap
import Tkinter as tk
from Tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------
# Last updated 10/22/2019 by aburky@princeton.edu
# --------------------------------------------------------------------------------------------------

# Define GUI
root = tk.Tk()


# Define GUI update function
def update():
    evdp = float(vardep.get())*1000
    source_lat = float(varlat.get())
    source_lon = float(varlon.get())
    rec_lat = float(varstla.get())
    rec_lon = float(varstlo.get())
    source = instaseis.Source(latitude=source_lat, longitude=source_lon, depth_in_m=evdp,
                              m_rr=m_rr, m_tt=m_tt, m_pp=m_pp, m_rt=m_rt, m_rp=m_rp, m_tp=m_tp,
                              origin_time=origin_time)
    st = db.get_seismograms(source=source, receiver=receiver, kind='displacement', dt=delta)
    # Get three component seismic data
    trz = st[0]
    trn = st[1]
    tre = st[2]
    # Calculate great circle distance and azimuth
    gcarc, azimuth = su.haversine(source_lat, source_lon, rec_lat, rec_lon)
    # Get P arrival time for windowing
    arrivals = model.get_travel_times(source_depth_in_km=evdp/1000, distance_in_degree=gcarc,
                                      phase_list=phases)
    p = arrivals[0]
    p410s = arrivals[1]
    p660s = arrivals[2]
    # Rotate data from N-E to R-T coordinate system
    trr, trt = su.seisrt(trn, tre, azimuth)
    # Get indices for cutting
    bidx = int(round((p.time - source_shift - 60) / delta) - 1)
    eidx = int(round((p.time - source_shift + 90) / delta))
    # Cut trace
    trrc = trr[bidx:eidx]
    trzc = trz[bidx:eidx]
    NT = np.size(trzc)
    window = signal.tukey(NT, alpha=0.25)
    trrc = trrc * window
    trzc = trzc * window
    # Calculate receiver function
    tshift = 10.0
    gw = 1.0
    nit = 1000
    errtol = 0.001
    [rf, rms] = rfunc.iterdecon(trrc, trzc, 1 / sample_rate, NT, tshift, gw, nit, errtol)

    # Plot everything!
    f = plt.figure(1, figsize=(12, 10))
    plt.clf()

    # Plot the radial trace
    p1 = f.add_subplot(221)
    p1.plot(t, trr, 'k', linewidth=0.5)
    p1.axvline(p.time, color='r', linewidth=0.25, label='P')
    p1.set_xlim(p.time - 60, p.time + 90)
    p1.set_ylim(np.min(trr[bidx:eidx]) * 1.2, np.max(trr[bidx:eidx]) * 1.2)
    p1.set_ylabel('Displacement (m)')
    p1.set_xlabel('Time after Origin (s)')
    p1.set_title(r"$u_R(t)$")
    p1.legend()

    # Plot the vertical trace
    p2 = f.add_subplot(223)
    p2.plot(t, trz, 'k', linewidth=0.5)
    p2.axvline(p.time, color='r', linewidth=0.25, label='P')
    p2.set_xlim(p.time - 60, p.time + 90)
    p2.set_ylim(np.min(trz[bidx:eidx]) * 1.2, np.max(trz[bidx:eidx]) * 1.2)
    p2.set_ylabel('Displacement (m)')
    p2.set_xlabel('Time after Origin (s)')
    p2.set_title(r"$u_Z(t)$")
    p2.legend()

    # Plot the receiver function
    p3 = f.add_subplot(224)
    p3.plot(trf, rf, 'k', linewidth=1)
    p3.axvline(tshift + (p410s.time - p.time), color='b', linewidth=0.25, label='P410s')
    p3.axvline(tshift + (p660s.time - p.time), color='g', linewidth=0.25, label='P660s')
    p3.set_xlim(np.min(trf), np.max(trf))
    p3.set_ylim(np.min(rf) * 1.1, np.max(rf) * 1.1)
    p3.set_title(r"$f_{Z \rightarrow R}(t)$")
    p3.set_xlabel('Time (s)')
    p3.legend()

    # Plot a map!
    p4 = f.add_subplot(222)
    m = Basemap(projection='robin', lon_0=0, resolution='c', ax=p4)
    m.drawcoastlines(linewidth=0.1)
    m.drawparallels(np.arange(-90.0, 120.0, 30.0), linewidth=0.2)
    m.drawmeridians(np.arange(0.0, 360.0, 60.0), linewidth=0.2)
    # Color things
    m.fillcontinents(color=(0.9, 0.9, 0.9), lake_color=(1, 1, 1))
    m.drawmapboundary(fill_color=(1, 1, 1))
    # Add event - station data
    m.drawgreatcircle(source_lon, source_lat, rec_lon, rec_lat, linewidth=1, color=(1, 0, 0))
    sx, sy = m(source_lon, source_lat)
    m.plot(sx, sy, '*', color=(0.9, 0.9, 0), markersize=18, markeredgecolor='black')
    rx, ry = m(rec_lon, rec_lat)
    m.plot(rx, ry, 'v', color=(1, 0, 0), markersize=12, markeredgecolor='black')

    # plt.subplots_adjust(hspace=0.35)
    f.canvas.draw()


# Set matplotlib to use LaTeX
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Open Instaseis database
db = instaseis.open_db("../Data/tiger-5s-iasp91-gauss-repack")
print(db)

# FIND WAY TO EXTRACT THIS AUTOMATICALLY
source_shift = 8.715

# Define moment tensor components
m_rr = 1.710000e+24 / 1E7
m_tt = 3.810000e+24 / 1E7
m_pp = -4.740000e+24 / 1E7
m_rt = 3.990000e+23 / 1E7
m_rp = -8.050000e+23 / 1E7
m_tp = -1.230000e+24 / 1E7

fm = (m_rr, m_tt, m_pp, m_rt, m_rp, m_tp)

# Define source and receiver parameters
source_lat = 70.0000
source_lon = 0.0000
source_dep_m = 300000
rec_lat = 40.0000
rec_lon = -80.0000
network = "IU"
station = "BBSR"
origin_time = obspy.UTCDateTime(2019, 1, 2, 3, 4, 5)

# Define output data parameters
delta = 0.5

# Call Instaseis and retrieve seismograms
receiver = instaseis.Receiver(latitude=rec_lat, longitude=rec_lon, network=network, station=station)
source = instaseis.Source(latitude=source_lat, longitude=source_lon, depth_in_m=source_dep_m,
                          m_rr=m_rr, m_tt=m_tt, m_pp=m_pp, m_rt=m_rt, m_rp=m_rp, m_tp=m_tp,
                          origin_time=origin_time)
st = db.get_seismograms(source=source, receiver=receiver, kind='displacement', dt=delta)
print(st)

# Save seismic traces into arrays
trz = st[0]
trn = st[1]
tre = st[2]

# GET P ARRIVAL TIME
# Define model
model = TauPyModel(model="iasp91")
# Define desired output parameters
phases = ["P", "P410s", "P660s"]

# Calculate great circle distance and azimuth
gcarc, azimuth = su.haversine(source_lat, source_lon, rec_lat, rec_lon)
# Rotate data from N-E to R-T coordinate system
trr, trt = su.seisrt(trn, tre, azimuth)

# Get arrival times for desired phases
arrivals = model.get_travel_times(source_depth_in_km=source_dep_m/1000, distance_in_degree=gcarc, phase_list=phases)
# Display P wave arrival time
p = arrivals[0]
# print(p.time)
p410s = arrivals[1]
# print(p410s.time - p.time)
p660s = arrivals[2]
# Make a plot showing ray paths
# paths = model.get_ray_paths(source_depth_in_km=source_dep_m/1000, distance_in_degree=gcarc, phase_list=phases)
# ax = paths.plot_rays()

# Construct time vector
t = trz.times() + 2*1.245
trf = np.arange(0, 150.5, delta)
sample_rate = 1/delta

# Get indices for cutting
bidx = int(round((p.time - source_shift - 60)/delta) - 1)
eidx = int(round((p.time - source_shift + 90)/delta))

# Filter trace?

# Cut trace
trrc = trr[bidx:eidx]
trzc = trz[bidx:eidx]
NT = np.size(trzc)
window = signal.tukey(NT, alpha=0.25)
trrc = trrc*window
trzc = trzc*window

# Calculate receiver function
tshift = 10.0
gw = 1.0
nit = 1000
errtol = 0.001
[rf, rms] = rfunc.iterdecon(trrc, trzc, 1/sample_rate, NT, tshift, gw, nit, errtol)

# Plot the traces and the receiver function
f = plt.figure(1, figsize=(14, 8))

# Plot the radial trace
p1 = f.add_subplot(221)
p1.plot(t, trr, 'k', linewidth=0.5)
p1.axvline(p.time, color='r', linewidth=0.25, label='P')
p1.set_xlim(p.time-60, p.time+90)
p1.set_ylim(np.min(trr[bidx:eidx])*1.2, np.max(trr[bidx:eidx])*1.2)
p1.set_ylabel('Displacement (m)')
p1.set_xlabel('Time after Origin (s)')
p1.set_title(r"$u_R(t)$")
p1.legend()

# Plot the vertical trace
p2 = f.add_subplot(223)
p2.plot(t, trz, 'k', linewidth=0.5)
p2.axvline(p.time, color='r', linewidth=0.25, label='P')
p2.set_xlim(p.time-60, p.time+90)
p2.set_ylim(np.min(trz[bidx:eidx])*1.2, np.max(trz[bidx:eidx])*1.2)
p2.set_ylabel('Displacement (m)')
p2.set_xlabel('Time after Origin (s)')
p2.set_title(r"$u_Z(t)$")
p2.legend()

# Plot the receiver function
p3 = f.add_subplot(224)
p3.plot(trf, rf, 'k', linewidth=1)
p3.axvline(tshift + (p410s.time - p.time), color='b', linewidth=0.25, label='P410s')
p3.axvline(tshift + (p660s.time - p.time), color='g', linewidth=0.25, label='P660s')
p3.set_xlim(np.min(trf), np.max(trf))
p3.set_ylim(np.min(rf)*1.1, np.max(rf)*1.1)
p3.set_title(r"$f_{Z \rightarrow R}(t)$")
p3.set_xlabel('Time (s)')
p3.legend()

# Plot a map!
p4 = f.add_subplot(222)
m = Basemap(projection='robin', lon_0=0, resolution='c', ax=p4)
m.drawcoastlines(linewidth=0.1)
m.drawparallels(np.arange(-90.0, 120.0, 30.0), linewidth=0.2)
m.drawmeridians(np.arange(0.0, 360.0, 60.0), linewidth=0.2)
# Color things
m.fillcontinents(color=(0.9, 0.9, 0.9), lake_color=(1, 1, 1))
# m.drawmapboundary(fill_color=(1, 1, 1))
m.drawmapboundary()
# Add event - station data
m.drawgreatcircle(source_lon, source_lat, rec_lon, rec_lat, linewidth=1, color=(1, 0, 0))
# sx, sy = m(source_lon, source_lat)
# m.plot(sx, sy, '*', color=(0.9, 0.9, 0), markersize=18, markeredgecolor='black')

# Add moment tensor as source
ax = plt.gca()
bb = obspy.imaging.beachball.beach(fm, linewidth=1, facecolor='k', xy=m(source_lon, source_lat), width=(2000000, 2000000), size=1000)
ax.add_collection(bb)

rx, ry = m(rec_lon, rec_lat)
m.plot(rx, ry, 'v', color=(1, 0, 0), markersize=14, markeredgecolor='black')

f.subplots_adjust(hspace=0.35)

# GUI details
canvas = FigureCanvasTkAgg(f, master=root)
root.title("InstaFunc")
plot_widget = canvas.get_tk_widget()

# Define variables which can be updated
vardep = StringVar()
varlat = StringVar()
varlon = StringVar()
varstla = StringVar()
varstlo = StringVar()

plot_widget.grid(row=0, column=0, columnspan=1, rowspan=8)
tk.Button(root, text="Run!", command=update, bg='green').grid(row=0, column=1, columnspan=1)
tk.Button(root, text="Quit", command=root.quit, bg='red').grid(row=0, column=2, columnspan=1)
# User entry interfaces
tk.Label(root, text='Event Latitude (deg)').grid(row=2, column=1)
tk.Entry(root, textvariable=varlat).grid(row=3, column=1)
tk.Label(root, text='Event Longitude (deg)').grid(row=2, column=2)
tk.Entry(root, textvariable=varlon).grid(row=3, column=2)
tk.Label(root, text='Event Depth (km)').grid(row=4, column=1)
tk.Entry(root, textvariable=vardep).grid(row=5, column=1)
tk.Label(root, text='Station Latitude (deg)').grid(row=6, column=1)
tk.Entry(root, textvariable=varstla).grid(row=7, column=1)
tk.Label(root, text='Station Longitude (deg)').grid(row=6, column=2)
tk.Entry(root, textvariable=varstlo).grid(row=7, column=2)

root.mainloop()
# plt.show()

# NEXT: Find sample indices corresponding to 60 seconds prior to and 90 seconds after theoretical P arrival time
#       for Instaseis data. This will require information about the sampling interval and the start time
#       - Think about the 'time_shift' that Instaseis implements and figure out robust timing
#       - S arrival time is not working as expected... maybe try on higher frequency database?
