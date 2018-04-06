using Base.Test
import divand

bathname = joinpath(dirname(@__FILE__),"..","..","divand-example-data","Global","Bathymetry","gebco_30sec_16.nc")
bathisglobal = true

if !isfile(bathname)
    bathname = download("https://b2drop.eudat.eu/s/o0vinoQutAC7eb0/download")
end

dx = dy = 0.1
dx = dy = 0.2

lonr = 27:dx:42
latr = 40:dy:47

depthr = [0.,10,100]

mask,(pm,pn,po),(xi,yi,zi) = divand.domain(bathname,bathisglobal,lonr,latr,depthr)


@test sum(mask[:,:,1]) >= sum(mask[:,:,2]) >= sum(mask[:,:,3])


mask,(pm,pn,po),(xi,yi,zi) = divand.domain(
    bathname,bathisglobal,
    lonr,latr,depthr;
    zlevel = :floor
)

# more than half of the points should be masked
# (in fact, for this domain, all points are masked at the boundary)
@test sum(mask[:,1,1]) < size(mask,1)/2



@test divand.localresolution([0.,10.,20.]) ≈ [10.,10.,10.]

@test divand.localresolution([0.,10.,20.,100,200,500]) ≈ [10.0,  10.0,  45.0,  90.0, 200.0, 300.0]



mask,pmn,xyi = divand.divand_rectdom(1:5,1:2:10,[10,20,30,100,200,300])


@test all(pmn[1] .== 1)
@test all(pmn[2] .== 1./2)
@test all(pmn[3][:,:,1] .== 1/10)
@test all(pmn[3][:,:,end] .== 1/100)