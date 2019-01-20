"""
Form the inverse of the background error covariance matrix.
s = DIVAnd_background(mask,pmn,Labs,alpha,moddim)
Form the inverse of the background error covariance matrix with
finite-difference operators on a curvilinear grid
# Input:
* mask: binary mask delimiting the domain. 1 is inside and 0 outside.
        For oceanographic application, this is the land-sea mask.
* pmn: scale factor of the grid.
* Labs: correlation length
* alpha: a dimensional coefficients for norm, gradient, laplacian,...
     alpha is usually [1,2,1] in 2 dimensions.
# Output:
*  s: stucture containing
    * s.iB: inverse of the background error covariance
    * s.L: spatial average correlation length
    * s.n: number of dimenions
    * s.coeff: scaling coefficient such that the background variance
         diag(inv(iB)) is one far away from the boundary.
"""
function DIVAnd_background(operatortype,mask,pmn,Labs,alpha,moddim,
                           scale_len = true,mapindex = []; kwargs...)
    Labs = len_harmonize(Labs,mask)
    return DIVAnd_background(operatortype,mask,pmn,Labs,alpha,moddim,scale_len,
                             mapindex; kwargs...)
end

function DIVAnd_background(operatortype,mask,pmn,
                           Labs::NTuple{N,AbstractArray{T,N}},alpha,moddim,
                           scale_len = true, mapindex = [];
                           btrunc = []) where {T,N}
    # number of dimensions
    n = ndims(mask)

    neff, alpha = alpha_default(Labs,alpha)

    sz = size(mask)

    if isempty(moddim)
        moddim = zeros(n)
    end

    iscyclic = moddim .> 0

    # scale iB such that the diagonal of inv(iB) is 1 far from
    # the boundary
    # we use the effective dimension neff to take into account that the
    # correlation length-scale might be zero in some directions


    coeff = 1.
    len_scale = 1.
    try
        coeff,K,len_scale = DIVAnd_kernel(neff,alpha)
    catch err
        if isa(err, DomainError)
            @warn "no scaling for alpha=$(alpha)"
        else
            rethrow(err)
        end
    end

    if scale_len
        # scale Labs by len_scale so that all kernels are similar
        Labs =
            let len_scale = len_scale, Labs = Labs
                ntuple(i -> Labs[i]/len_scale,Val(N))
            end
    end

    # mean correlation length in every dimension
    Ld = [mean(L) for L in Labs]
    neff = sum(Ld .> 0)

    # geometric mean
    geomean(v) = prod(v)^(1/length(v))
    L = geomean(Ld[Ld .> 0])

    alphabc = 0

    Labs2 =
        let Labs = Labs
            ntuple(i -> Labs[i].^2,Val(N))
        end

    s,D = DIVAnd_operators(operatortype,mask,pmn,Labs2,
                           iscyclic,mapindex,Labs)

    # D is laplacian (a dimensional, since nu = Labs.^2)

    # norm taking only dimension into account with non-zero correlation
    # WE: units length^(neff/2)

    ivol_eff = .*(pmn[Ld .> 0]...)

	WE = oper_diag(operatortype,statevector_pack(s.sv,(1 ./ sqrt.(ivol_eff),))[:,1])

	Ln = prod(Ld[Ld .> 0])

	coeff = coeff * Ln # units length^n

    pmnv = zeros(eltype(pmn[1]),length(mask),N)
    for i = 1:N
        if Ld[i] == 0
            pmnv[:,i] .= 1
        else
            pmnv[:,i] = pmn[i][:]
        end
    end

	for i=1:N
        # staggered version of norm
		S = sparse_stagger(sz,i,iscyclic[i])
		ma = (S * mask[:]) .== 1

        spack = sparse_pack(ma)

        # d = @static if VERSION >= v"0.7.0-beta.0"
		#     #spack * (prod(S * pmnv,dims=2)[:,1])
        #     spack * S * view(ivol_eff,:)
        # else
		#     spack * (prod(S * pmnv,2)[:,1])
        # end
        d = spack * (S * ivol_eff[:])

        for j = 1:length(d)
            d[j] = sqrt(1/d[j])
        end
		s.WEs[i] = oper_diag(operatortype,d)

        # staggered version of norm scaled by length-scale
		Li2 = Labs[i][:].^2

		tmp = spack * sqrt.(S*Li2[:])
		s.WEss[i] = oper_diag(operatortype,tmp) * s.WEs[i]
	end

    # adjust weight of halo points
	if !isempty(mapindex)
		# ignore halo points at the center of the cell

		WE = oper_diag(operatortype,s.isinterior) * WE

        # divide weight be two at the edged of halo-interior cell
        # weight of the grid points between halo and interior points
        # are 1/2 (as there are two) and interior points are 1
		for i=1:n
			s.WEs[i] = oper_diag(operatortype,sqrt.(s.isinterior_stag[i])) * s.WEs[i]
		end
	end

	s.WE = WE
	s.coeff = coeff
	# number of dimensions
	s.n = N

	# mean correlation legth
	s.Ld = Ld
    #JLD2.@save "/tmp/DIVAnd_background_components.jld2"  s D alpha btrunc
	iB = DIVAnd_background_components(s,D,alpha,btrunc=btrunc)

	# inverse of background covariance matrix
	s.iB = iB

	#s.Ln = Ln

	s.moddim = moddim
	s.iscyclic = iscyclic

	s.alpha = alpha
	s.neff = neff
	s.WE = WE # units length^(n/2)

	return s
end

# Copyright (C) 2014, 2017 Alexander Barth 		<a.barth@ulg.ac.be>
#                         Jean-Marie Beckers 	<jm.beckers@ulg.ac.be>
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
