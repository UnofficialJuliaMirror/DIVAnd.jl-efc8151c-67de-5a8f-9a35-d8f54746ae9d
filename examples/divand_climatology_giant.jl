#SBATCH --mem-per-cpu=16000

#ENV["MEMTOFIT"]=8.0

using divand
using PyPlot
using NetCDF

function loadbigfile(fname)

    data = readlines(open(fname,"r"))
    nobs = length(data)

    lon = zeros(nobs)
    lat = zeros(nobs)
    depth = zeros(nobs)
    time = Array{DateTime}(nobs)
    value = zeros(nobs)
    id = Array{String}(nobs)


    for i in 1:nobs
        rec = split(data[i])
        lon[i] = parse(Float64,rec[1])
        lat[i] = parse(Float64,rec[2])
        value[i] = parse(Float64,rec[3])
        depth[i] = parse(Float64,rec[4])
        time[i] = DateTime(rec[10])
        id[i] = rec[11]
    end

    return value,lon,lat,depth,time,id
end

include("../src/override_ssmult.jl")

# if this script is in /some/path/divand.jl/examples, the data should be in
# /some/path/divand-example-data (for Linux, Mac) and likewise for Windows.
fname = joinpath(dirname(@__FILE__),"..","..","divand-example-data","BlackSea","Salinity.bigfile")
bathname = joinpath(dirname(@__FILE__),"..","..","divand-example-data","Global","Bathymetry","gebco_30sec_16.nc")

isglobal = true

if !isdefined(:value)
    value,lon,lat,depth,time,id = loadbigfile(fname)
end

@show size(value)

dx = dy = 0.1
dx = dy = 0.2
dx = dy = 0.07
#dx = dy = 0.04
#dx = dy = 0.03
dx=15.0/(50*16)
dy=6.0/(15*16)

lonr = 27:dx:42
latr = 40.4:dy:46.6

lonr = 27:dx:42
latr = 40:dy:47


#depthr = [0., 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000];
depthr = [0.,5, 10, 15, 20, 25, 30, 40, 50, 66, 75, 85, 100, 112, 125, 135, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1600, 1750, 1850, 2000];
@show size(depthr)
#depthr = [0, 10, 20, 30, 50, 75, 100];
#depthr = 0:10.:30.;


timer = 1:1.:12

mask,(pm,pn,po,pp),(xi,yi,zi,ti) = divand_rectdom(lonr,latr,depthr,timer)
@show size(mask)     ()

epsilon2 = 0.5

time2 = Dates.month.(time)

mxi,myi,mask2 = load_mask(bathname,isglobal,minimum(lonr),maximum(lonr),dx,minimum(latr),maximum(latr),dy,depthr)

mask3 = repeat(mask2,inner = (1,1,1,length(timer)))
#only see points
#mask3 = mask

sz = size(mask)
@show sz
@show size(mask3)

z = zeros(sz)
# correlation length in arc degree
#lenx = fill(4*dx,sz)
lenx=4*dx
#leny = fill(4*dy,sz)
leny=4*dy
# correlation length in meters
#lenz = (10 + zi/5)/3
lenz=3
@show mean(lenz)
# correlation time-scale in month
#lent = fill(1.,sz)
lent=1.

moddim=[0,0,0,12]

# Prepare background as mean vertical profile and time evolution. Just call divand in two dimensions forgetting x and y ...

vm=mean(value)
va=value-vm

toaverage=[true true false false]

@time fi,ffb=divand_averaged_bg(mask3,(pm,pn,po,pp),(xi,yi,zi,ti),(lon,lat,depth,time2),va,(lenx,leny,4*lenz,4*lent),epsilon2*10,toaverage;moddim=moddim)

# save background field
divand_save(replace(@__FILE__,r".jl$","hr0.02SB.nc"),mask,"Salinity",fi+vm)

# try to clean up 
fi=0

z=0


vaa=va-ffb

fi,erri=divandgo(mask3,(pm,pn,po,pp),(xi,yi,zi,ti),(lon,lat,depth,time2),vaa,(lenx,leny,lenz,lent),epsilon2,:none;moddim=moddim,MEMTOFIT=3.0)


#fi=fi+fmb+vm
# Why is this filter necessary; sharedArray not supported ??
#fi=divand_filter3(fi,NaN,2)
#erri=divand_filter3(erri,NaN,2)

# Save anomalies 
divand_save(replace(@__FILE__,r".jl$","hr0.02S.nc"),mask,"Salinity",fi)

#divand_save(replace(@__FILE__,r".jl$","hr0.02E.nc"),mask,"Errorfield",erri)
nothing