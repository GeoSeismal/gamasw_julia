using Plots,ForwardDiff,LinearAlgebra,LaTeXStrings,Interpolations,BenchmarkTools,DelimitedFiles

function dc_inversion(freq_obs,c_obs,Vp,Vs,ρ,d,cmin,cmax; dc=25.0,maxiter=10,
    isverbose=false, gradientmode="AD", α=1e-2, α_multiplier = 0.8, n_linesearch = 20,
    iterverbose=5, alphamode="constant")
c_array = range(cmin,cmax+dc,step=dc)
misfit = []
push!(misfit,calculate_L2norm_misfit(c_obs,Vp, Vs, ρ, d, freq_obs, c_array))
_Vs = copy(Vs)
iter = 1
isdecrease = false
# global α = copy(α)
global _α = copy(α)
global _misfit = misfit[end]
global grad_f_Vs_old = zeros(size(Vs))
global grad_f_Vs = zeros(size(Vs))
global Δm = zeros(size(Vs))
global Δm_old = zeros(size(Vs))
while iter<=maxiter
    if isverbose && (iter-1)%iterverbose==0; println("Iteration $(iter) with misfit $(misfit[end]) and alpha $(α);\t"); end
    if iter==1
        if gradientmode=="FD"
            grad_f_Vs = partial_misfit_wrt_Vs_FD(c_obs,Vp, _Vs, ρ, d, freq_obs, c_array)
        else
            grad_f_Vs = partial_misfit_wrt_Vs(c_obs,Vp, _Vs, ρ, d, freq_obs, c_array)
        end
        Δm = -grad_f_Vs
    else
        if gradientmode=="FD"
            grad_f_Vs = partial_misfit_wrt_Vs_FD(c_obs,Vp, _Vs, ρ, d, freq_obs, c_array)
        else
            grad_f_Vs = partial_misfit_wrt_Vs(c_obs,Vp, _Vs, ρ, d, freq_obs, c_array)
        end
        beta = (transpose(grad_f_Vs)*(grad_f_Vs-grad_f_Vs_old))/(transpose(grad_f_Vs)*grad_f_Vs)
        Δm = -grad_f_Vs + beta.*Δm_old
    end

    if alphamode == "backtracking"
        # backtracking line-search
        try
            i_linesearch = 1
            _α = copy(α)
            isnotcompute = true
            _tmpVs = _Vs .+ _α.*Δm
            _misfit = calculate_L2norm_misfit(c_obs,Vp, _tmpVs, ρ, d, freq_obs, c_array)
            # try to increase α
            
            while _misfit<misfit[end] && i_linesearch<= n_linesearch #&& ~isdecrease
                try
                    α = copy(_α)
                    _α = _α/α_multiplier
                    _tmpVs = _Vs .+ _α.*Δm
                    _misfit = calculate_L2norm_misfit(c_obs,Vp, _tmpVs, ρ, d, freq_obs, c_array)
                    i_linesearch += 1
                catch
                    if i_linesearch ==1; isdecrease = true; end
                end
            end
            
            if _misfit>misfit[end] && ~isdecrease
                if i_linesearch ==1; isdecrease = true; end
            end

            # if fails, try to decrease α
            while _misfit>misfit[end] && i_linesearch<= n_linesearch && isdecrease
                try
                    α = copy(_α)
                    _α = _α * α_multiplier
                    _tmpVs = _Vs .+ _α.*Δm
                    _misfit = calculate_L2norm_misfit(c_obs,Vp, _tmpVs, ρ, d, freq_obs, c_array)
                    # _α = _α * α_multiplier
                    i_linesearch += 1
                catch
                    nothing
                end
                α = copy(_α)
            end
        catch
            α = copy(α)
        end
    elseif alphamode=="decreasing"
        try
            _α = copy(α)
            i_linesearch = 1
            _tmpVs = _Vs .+ _α.*Δm
            _misfit = calculate_L2norm_misfit(c_obs,Vp, _tmpVs, ρ, d, freq_obs, c_array)

            while _misfit>misfit[end] && i_linesearch<=n_linesearch
                _α = _α * α_multiplier
                _tmpVs = _Vs .+ _α.*Δm
                _misfit = calculate_L2norm_misfit(c_obs,Vp, _tmpVs, ρ, d, freq_obs, c_array)
                i_linesearch += 1
                println("i_linesearch : ",i_linesearch)
            end
        catch
            nothing
        end
        α = copy(_α)
    elseif alphamode=="constant"
        nothing
    end
    if isverbose && (iter-1)%iterverbose==0;println("α = ",α," misfit: ",misfit[end]," _misfit: ",_misfit);end

    _Vs = _Vs .+ α.*Δm
    grad_f_Vs_old = copy(grad_f_Vs)
    Δm_old = copy(Δm)
    push!(misfit,calculate_L2norm_misfit(c_obs,Vp, _Vs, ρ, d, freq_obs, c_array))
    iter += 1
end
return _Vs, misfit
end