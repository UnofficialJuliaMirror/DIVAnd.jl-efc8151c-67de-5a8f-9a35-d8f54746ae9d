# Testing divand in 2 dimensions

using Base.Test
#using divand

# grid of background field
xi,yi = ndgrid(linspace(0,1,30),linspace(0,1,30))

mask = trues(size(xi))
pm = ones(size(xi)) / (xi[2,1]-xi[1,1])
pn = ones(size(xi)) / (yi[1,2]-yi[1,1])

epsilon = 1e-10;

# grid of observations
x,y = ndgrid(linspace(epsilon,1-epsilon,20),linspace(epsilon,1-epsilon,20))
x = x[:]
y = y[:]
v = sin(x*6) .* cos(y*6)


lenx = .15;
leny = .15;

epsilon2 = 1;

# direct inversion
@time va_chol,s = divandrun(mask,(pm,pn),(xi,yi),(x,y),v,(lenx,leny),epsilon2,inversion=:chol)

# iterative (without preconditioner)
@time va_iter,s = divandrun(mask,(pm,pn),(xi,yi),(x,y),v,(lenx,leny),epsilon2,inversion=:iter)
#@test va_chol ≈ va_iter
@show s.niter

# iterative (with preconditioner)
function compPC(iB,H,R) 
    @show "hi"
    return x -> x;
end

va_iter,s = divandrun(mask,(pm,pn),(xi,yi),(x,y),v,(lenx,leny),epsilon2,inversion=:iter,compPC = compPC);
#@test va_chol ≈ va_iter
@show s.niter


va_iter,s = divandrun(mask,(pm,pn),(xi,yi),(x,y),v,(lenx,leny),epsilon2,inversion=:iter,compPC = divand_pc_sqrtiB);
#@test va_chol ≈ va_iter
@show s.niter



# Copyright (C) 2014, 2016 Alexander Barth <a.barth@ulg.ac.be>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <http://www.gnu.org/licenses/>.
