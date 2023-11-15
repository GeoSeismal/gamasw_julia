using Plots,ForwardDiff,LinearAlgebra,LaTeXStrings,Interpolations,BenchmarkTools,DelimitedFiles

function dispersion_function(ω, c, α, β, ρ, d; swtype="rayleigh", array_eltype=Float64)
	nd = length(d)

	X = zeros(array_eltype,(5))

	Cα = zeros(array_eltype,(nd,1))
	Sα = zeros(array_eltype,(nd,1))
	Cβ = zeros(array_eltype,(nd,1))
	Sβ = zeros(array_eltype,(nd,1))
	r = zeros(array_eltype,(nd,1))
	s = zeros(array_eltype,(nd,1))

	ϵ = zeros(array_eltype,(nd-1,1))
	η = zeros(array_eltype,(nd-1,1))
	a = zeros(array_eltype,(nd-1,1))
	ap = zeros(array_eltype,(nd-1,1))
	b = zeros(array_eltype,(nd-1,1))
	bp = zeros(array_eltype,(nd-1,1))

	Tl = zeros(array_eltype,(2,2))
	Tli = zeros(array_eltype,(2,2,nd-1))
	U = zeros(array_eltype,(2))
	V = zeros(array_eltype,(2))
	k = ω/c
	μ = ρ.*β.^2
	for i in range(1,nd)
		if swtype=="rayleigh"
			if isless(c,α[i])
				r[i] = sqrt.(1.0 .- c.^2 ./α[i].^2)
				Cα[i] = cosh(k * r[i] * d[i])
				Sα[i] = sinh(k * r[i] * d[i])
			elseif c > α[i]
				r[i] = sqrt.(c.^2 ./ α[i].^2 .- 1.0)
				Cα[i] = cos(k * r[i] * d[i])
				Sα[i] = sin(k * r[i] * d[i])
			else
				r[i] = 0.0
				Cα[i] = 1.0
				Sα[i] = 0.0
			end
		end
		if c < β[i]
			s[i] = sqrt(1.0 .- c.^2 ./ β[i].^2)
			Cβ[i] = cosh(k * s[i] * d[i])
			Sβ[i] = sinh(k * s[i] * d[i])
		elseif c > β[i]
			s[i] = sqrt(c.^2 ./ β[i].^2 .- 1.0)
			Cβ[i] = cos(k * s[i] * d[i])
			Sβ[i] = sin(k * s[i] * d[i])
		else
			s[i] = 0.0
			Cβ[i] = 1.0
			Sβ[i] = 0.0
		end
	end
	if swtype=="rayleigh"
		γ = β.^2 ./ c.^2
		t = 2.0 .- c.^2 ./ β.^2
		for i in range(1,nd-1)
			ϵ[i] = ρ[i+1]./ρ[i]
			η[i] = 2.0 .* (γ[i] .- ϵ[i] .* γ[i + 1])
		end
		a = ϵ .+ η
		ap = a .- 1.0
		b = 1.0 .- η
		bp = b .- 1.0
		_X = μ[1].^2 .* [2.0*t[1] -t[1].^2 0.0 0.0 -4.0]
		for i in range(1,nd-1)
			if c < β[i]
				p1 = Cβ[i] * _X[2] + s[i] * Sβ[i] * _X[3]
				p2 = Cβ[i] * _X[4] + s[i] * Sβ[i] * _X[5]
			else
				p1 = Cβ[i] * _X[2] - s[i] * Sβ[i] * _X[3]
				p2 = Cβ[i] * _X[4] - s[i] * Sβ[i] * _X[5]
			end
			if c ≠ β[i]
				p3 = 1.0 / s[i] * Sβ[i] * _X[2] + Cβ[i] * _X[3]
				p4 = 1.0 / s[i] * Sβ[i] * _X[4] + Cβ[i] * _X[5]
			else
				p3 = k * d[i] * _X[2] + Cβ[i] * _X[3]
				p4 = k * d[i] * _X[4] + Cβ[i] * _X[5]
			end
			if c < α[i]
				q1 = Cα[i] * p1 - (r[i] * Sα[i]) * p2
				q3 = Cα[i] * p3 - (r[i] * Sα[i]) * p4
			else
				q1 = Cα[i] * p1 + (r[i] * Sα[i]) * p2
				q3 = Cα[i] * p3 + (r[i] * Sα[i]) * p4
			end
			if c ≠ α[i]
				q2 = -1.0 / r[i] * Sα[i] * p3 + Cα[i] * p4
				q4 = -1.0 / r[i] * Sα[i] * p1 + Cα[i] * p2
			else
				q2 = -k * d[i] * p3 + Cα[i] * p4
				q4 = -k * d[i] * p1 + Cα[i] * p2
			end
			y1 = ap[i] * _X[1] + a[i] * q1
			y2 = a[i] * _X[1] + ap[i] * q2
			z1 = b[i] * _X[1] + bp[i] * q1
			z2 = bp[i] * _X[1] + b[i] * q2

			_X[1] = bp[i] * y1 + b[i] * y2
			_X[2] = a[i] * y1 + ap[i] * y2
			_X[3] = ϵ[i] * q3
			_X[4] = ϵ[i] * q4
			_X[5] = bp[i] * z1 + b[i] * z2
		end
		panel = _X[2] .+ s[nd] .* _X[3] .- r[nd] .* (_X[4] .+ s[nd] .* _X[5])
	elseif swtype=="love"
		Tli[1,1,:] = Cβ[1:nd-1]
		for i in range(1,nd-1)
			if c ≠ β[i]
				Tli[1,2,i] = Sβ[i]/μ[i]/s[i]
			else
				Tli[1,2,i] = k * d[i] / μ[i]
			end
		end
		Tli[2,1,:] = μ[1:nd-1] .* s[1:nd-1] .* Sβ[1:nd-1]
		Tli[2,2,:] = Cβ[1:nd-1]

		U = [0.0 1.0]
		V = [1.0 μ[nd]*s[nd]]
		Tl = Tli[:,:,1]
		for i in range(2,nd-1)
			_Tl = copy(Tl)
			_Tl = Tl * Tli[:,:,i]
		end
		panel = U*Tl ⋅ V
	end
	return panel
end

function dispersion_function_bridge(ω, c, α, β, ρ, d; swtype="rayleigh", mode="normal")
	if mode=="normal"
		panel = dispersion_function(ω, c, α, β, ρ, d; swtype=swtype, array_eltype=Float64)
	elseif mode=="α"
		panel = dispersion_function(ω, c, α, β, ρ, d; swtype=swtype, array_eltype=eltype(α))
	elseif mode=="β"
		panel = dispersion_function(ω, c, α, β, ρ, d; swtype=swtype, array_eltype=eltype(β))
	elseif mode=="ρ"
		panel = dispersion_function(ω, c, α, β, ρ, d; swtype=swtype, array_eltype=eltype(ρ))
	else
		error("Mode is not supported! The supported modes are: normal, α, β, or ρ.")
	end
	return panel
end

function fcpanel(f,c, α, β, ρ, d; swtype="rayleigh")
	nd = length(d)
	nc = length(c)
	nf = length(f)
	panel = zeros(eltype(β),(nc,nf))
	for i1 in range(1,nf)
		ω = 2*π*f[i1]
		for i2 in range(1,nc)
			# panel[i2,i1] = dispersion_function(ω, c[i2], α, β, ρ, d, swtype=swtype)
            panel[i2,i1] = dispersion_function_bridge(ω, c[i2], α, β, ρ, d; swtype=swtype, mode="β")
		end
	end
	return panel
end

function mode_extraction(fmap,f_array,c_array;maxmode=1,verbose=false)
	_fmap = copy(fmap)
	c_dc = [[] for i in range(1,maxmode)]
	f_dc = [[] for i in range(1,maxmode)]
	for i1 in range(1,size(fmap,1))
		_line = _fmap[i1,:]
		im = 1
		for (i,il) in enumerate(_line)
			_line = _line / maximum(abs.(_line))
			if i>1
				sign_i = sign(il)
				sign_im1 = sign(_line[i-1])
				if sign_i != sign_im1
					if verbose
						println("root is found at f ",f_array[i1] )
					end
					x_r = [_line[i-1] _line[i]]
					v_r = [c_array[i-1] c_array[i]]
					x = ((v_r[1]*x_r[2]) - (v_r[2]*x_r[1])) / (x_r[2] - x_r[1])
					f = f_array[i1]
					push!(c_dc[im],x)
					push!(f_dc[im],f)
					im = im + 1
					if im > maxmode
						break
					end
				end
			end
		end
	end
	return (c_dc,f_dc)
end

function mode_extraction_c(fmap,f_array,c_array;maxmode=1,verbose=false)
	_fmap = copy(fmap)
	c_dc = [[] for i in range(1,maxmode)]
	f_dc = [[] for i in range(1,maxmode)]
	for i1 in range(1,size(fmap,1))
		_line = _fmap[i1,:]
		im = 1
		for (i,il) in enumerate(_line)
			_line = _line / maximum(abs.(_line))
			if i>1
				sign_i = sign(il)
				sign_im1 = sign(_line[i-1])
				if sign_i != sign_im1
					if verbose
						println("root is found at f ",f_array[i1] )
					end
					x_r = [_line[i-1] _line[i]]
					v_r = [c_array[i-1] c_array[i]]
					x = ((v_r[1]*x_r[2]) - (v_r[2]*x_r[1])) / (x_r[2] - x_r[1])
					f = f_array[i1]
					push!(c_dc[im],x)
					push!(f_dc[im],f)
					im = im + 1
					if im > maxmode
						break
					end
				end
			end
		end
	end
	return c_dc
end

function calculate_dc_obs(Vp, Vs, ρ, d, 
					fmin, fmax, cmin, cmax; df=0.1, dc=100.0, maxmode=1)
	f_array = range(fmin,fmax+df,step=df)
	c_array = range(cmin,cmax+dc,step=dc)
	fmap = fcpanel(f_array,c_array, Vp, Vs, ρ, d, swtype="rayleigh")
	(phasevel,freq) = mode_extraction(transpose(fmap),f_array,c_array;maxmode=maxmode)
	return (phasevel,freq)
end

function calculate_dc_obs_c(Vp, Vs, ρ, d, 
					f_array,c_array; maxmode=1)
	fmap = fcpanel(f_array,c_array, Vp, Vs, ρ, d, swtype="rayleigh")
	phasevel = mode_extraction_c(transpose(fmap),f_array,c_array;maxmode=maxmode)
	return phasevel
end

function dividing_parameter(Vs,d;nlayer::Int64=15)
	Vp = Vs.*sqrt(3)
	ρ = 310.0.*Vp.^0.25
	
	d_tmp = d[1:length(d)-1]
	nh = nlayer
	layerthickness = sum(d_tmp)/(nlayer-1)
	layerdepth = cumsum(d_tmp)
	Vs_h = zeros(Float64,(nh,1))
	Vp_h = zeros(Float64,(nh,1))
	ρ_h = zeros(Float64,(nh,1))
	d_h = zeros(Float64,(nh,1))
	ilayer = 1
	for ih=1:nh
		if ih*layerthickness>layerdepth[ilayer]
			ilayer += 1
		end
		Vp_h[ih] = Vp[ilayer]
		Vs_h[ih] = Vs[ilayer]
		ρ_h[ih] = ρ[ilayer]
		d_h[ih] = layerthickness
	end
	return Vp_h,Vs_h,ρ_h,d_h
end

calc_Vp(Vs) = Vs.*sqrt(3)

calc_ρ(Vp) = 310.0*Vp.^0.25

function estimate_initial_model(f_obs,c_obs;nlayer=10)
	maximum_depth = 0.5*c_obs[argmin(f_obs)]/minimum(f_obs) # half of the wavelength of the observed phase velocity
	minimum_Vs = 1.1*minimum(c_obs)
	maximum_Vs = 2.0*maximum(c_obs)
    _Vs = [minimum_Vs+(maximum_Vs-minimum_Vs)*i/nlayer for i in range(1,nlayer)]
    _thickness = [maximum_depth/nlayer for i in range(1,nlayer)]
	return _Vs,_thickness
end

function calculate_L2norm_misfit(d_obs,Vp, Vs, ρ, d, f_array, c_array)
    d_cal = calculate_dc_obs_c(Vp, Vs, ρ, d, f_array, c_array;maxmode=1)
    return sum((d_obs.-d_cal[1]).^2)/length(d_obs)
end

partial_misfit_wrt_Vs(c_obs,Vp, Vs, ρ, d, f_array, c_array) = ForwardDiff.gradient(z -> calculate_L2norm_misfit(c_obs, Vp, z, ρ, d, f_array, c_array), Vs)

function partial_misfit_wrt_Vs_FD(c_obs,Vp, Vs, ρ, d, f_array, c_array; h=1.0)
    out = zeros(eltype(Vs),length(Vs))
    for i in range(1,length(Vs))
        _tmpVsplus = copy(Vs)
        _tmpVsplus[i] += h
        _tmpVsminus = copy(Vs)
        _tmpVsminus[i] -= h
        out[i] = (0.5*calculate_L2norm_misfit(c_obs,Vp, _tmpVsplus, ρ, d, f_array, c_array) - 
            0.5*calculate_L2norm_misfit(c_obs,Vp, _tmpVsminus, ρ, d, f_array, c_array))/h
    end
    return out
end

# function dc_inversion(freq_obs,c_obs,Vp,Vs,ρ,d,cmin,cmax; dc=25.0,maxiter=10,
#         isverbose=false, gradientmode="AD", α=1e-2, α_multiplier = 0.8, n_linesearch = 20,
#         iterverbose=5, alphamode="constant")
#     c_array = range(cmin,cmax+dc,step=dc)
#     misfit = []
#     push!(misfit,calculate_L2norm_misfit(c_obs,Vp, Vs, ρ, d, freq_obs, c_array))
#     _Vs = copy(Vs)
#     iter = 1
#     isdecrease = false
#     # global α = copy(α)
#     global _α = copy(α)
#     global _misfit = misfit[end]
#     global grad_f_Vs_old = zeros(size(Vs))
#     global grad_f_Vs = zeros(size(Vs))
#     global Δm = zeros(size(Vs))
#     global Δm_old = zeros(size(Vs))
#     while iter<=maxiter
#         if isverbose && (iter-1)%iterverbose==0; println("Iteration $(iter) with misfit $(misfit[end]) and alpha $(α);\t"); end
#         if iter==1
#             if gradientmode=="FD"
#                 grad_f_Vs = partial_misfit_wrt_Vs_FD(c_obs,Vp, _Vs, ρ, d, freq_obs, c_array)
#             else
#                 grad_f_Vs = partial_misfit_wrt_Vs(c_obs,Vp, _Vs, ρ, d, freq_obs, c_array)
#             end
#             Δm = -grad_f_Vs
#         else
#             if gradientmode=="FD"
#                 grad_f_Vs = partial_misfit_wrt_Vs_FD(c_obs,Vp, _Vs, ρ, d, freq_obs, c_array)
#             else
#                 grad_f_Vs = partial_misfit_wrt_Vs(c_obs,Vp, _Vs, ρ, d, freq_obs, c_array)
#             end
#             beta = (transpose(grad_f_Vs)*(grad_f_Vs-grad_f_Vs_old))/(transpose(grad_f_Vs)*grad_f_Vs)
#             Δm = -grad_f_Vs + beta.*Δm_old
#         end

#         if alphamode != "constant"
#             # backtracking line-search
#             try
#                 i_linesearch = 1
#                 _α = copy(α)
#                 isnotcompute = true
#                 _tmpVs = _Vs .+ _α.*Δm
#                 _misfit = calculate_L2norm_misfit(c_obs,Vp, _tmpVs, ρ, d, freq_obs, c_array)
#                 # try to increase α
                
#                 while _misfit<misfit[end] && i_linesearch<= n_linesearch #&& ~isdecrease
#                     try
#                         α = copy(_α)
#                         _α = _α/α_multiplier
#                         _tmpVs = _Vs .+ _α.*Δm
#                         _misfit = calculate_L2norm_misfit(c_obs,Vp, _tmpVs, ρ, d, freq_obs, c_array)
#                         i_linesearch += 1
#                     catch
#                         if i_linesearch ==1; isdecrease = true; end
#                     end
#                 end
                
#                 if _misfit>misfit[end] && ~isdecrease
#                     if i_linesearch ==1; isdecrease = true; end
#                 end
    
#                 # if fails, try to decrease α
#                 while _misfit>misfit[end] && i_linesearch<= n_linesearch && isdecrease
#                     try
#                         α = copy(_α)
#                         _α = _α * α_multiplier
#                         _tmpVs = _Vs .+ _α.*Δm
#                         _misfit = calculate_L2norm_misfit(c_obs,Vp, _tmpVs, ρ, d, freq_obs, c_array)
#                         # _α = _α * α_multiplier
#                         i_linesearch += 1
#                     catch
#                         nothing
#                     end
#                     α = copy(_α)
#                 end
#             catch
#                 α = copy(α)
#             end
#         else
#             nothing
#         end
#         # if isverbose && (iter-1)%iterverbose==0;println("α = ",α," misfit: ",misfit[end]," _misfit: ",_misfit);end

#         _Vs = _Vs .+ α.*Δm
#         grad_f_Vs_old = copy(grad_f_Vs)
#         Δm_old = copy(Δm)
#         push!(misfit,calculate_L2norm_misfit(c_obs,Vp, _Vs, ρ, d, freq_obs, c_array))
#         iter += 1
#     end
#     return _Vs, misfit
# end