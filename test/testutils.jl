#######################################
# CompetingSamplers
#######################################

# compute a row of (reversible) MetropolizedGibbs
function MetroGibbs(gs, idx)
    log1mexpgi = log1mexp(gs[idx])
    mets = map(g ->  g - max(log1mexp(g), log1mexpgi), gs)
    mets[idx] = -Inf
    mets[idx] = log1mexp(min(0.,logsumexp(mets)))
    exp.(mets)
end

# split the transition Matrix of a skew-detail sampler
function splitTransMat(P::Matrix)
    nlvls = size(P,1)÷2
    T⁺ = P[1:nlvls,1:nlvls]
    T⁻ = P[(nlvls+1):2nlvls,(nlvls+1):2nlvls]
    Λ⁺ = diag(P[1:nlvls,(nlvls+1):2nlvls])
    Λ⁻ = diag(P[(nlvls+1):2nlvls, 1:nlvls])
    return T⁺, T⁻, Λ⁺, Λ⁻
end

# create a transition matrix for (i,eps) and fixed x=xmin
function buildTransMat(sh::SH16Sampler)
    N       = sh.np.N
    nlvls   = N+1
    nstates = 2nlvls
    Prows   = [zeros(nstates) for _ in 1:nstates];
    for ieps in 1:2
        # ieps=1
        o = ieps == 1 ? 0 : nlvls
        sh.ip[2] = ieps == 1 ? 1 : -1
        for ii in 1:nlvls
            # ii = 1
            pidx = o+ii
            sh.ip[1] = ii-1
            iprop, lpm = NRSTExp.CompetingSamplers.propose_i(sh)
            Prows[pidx][o+iprop+1] = exp(lpm)
            lpflip = NRSTExp.CompetingSamplers.get_lpflip!(sh, lpm)
            sh.ip[2] *= -1               # need to undo flip inside get_lpflip!
            lpff   = log1mexp(lpm)
            Λ      = exp(lpflip + lpff)  # lpflip = log(Lambda) - log(1-exp(lpm)) <=> log(Lambda) = lpflip + log1mexp(lpm)
            Prows[pidx][nlvls-o+ii] = Λ
            Prows[pidx][pidx] = max(0., min(1., exp(lpff) - Λ))
        end
    end
    P = collect(hcat(Prows...)')
    return P
end

# create a transition matrix for (i,eps) and fixed x=xmin
function buildTransMat(fbdr::FBDRSampler)
    N       = fbdr.np.N
    nlvls   = N+1
    nstates = 2nlvls
    Prows   = [zeros(nstates) for _ in 1:nstates];
    for ieps in 1:2
        # ieps=1
        o = ieps == 1 ? 0 : nlvls
        fbdr.ip[2] = ieps == 1 ? 1 : -1
        for ii in 1:nlvls
            # ii = 1
            pidx = o+ii
            fbdr.ip[1] = ii-1
            NRSTExp.CompetingSamplers.update_ms!(fbdr)
            lpff = log1mexp(min(0., logsumexp(fbdr.ms)))

            # move i
            msidxs = ieps==1 ? ((ii+1):nlvls) : (1:(ii-1))
            jdxs   = o .+ msidxs
            if length(jdxs) > 0
                Prows[pidx][jdxs] .= exp.(fbdr.ms[msidxs])
            end

            # flip eps
            fbdr.ip[2] *= -1                             # simulate flip
            NRSTExp.CompetingSamplers.update_ms!(fbdr)   # recompute IMGS probabilities
            lpfb = log1mexp(min(0., logsumexp(fbdr.ms))) # logprob of failing to sample from {j: (j-i)eps'>0} with the flipped eps'=-eps. also, need to trunc to avoid issues with numerical noise
            Λ    = max(0., exp(lpff) - exp(lpfb))        # exp(a) - exp(b) = exp(b)(exp(a-b)-1)=exp(b)expm1(a-b)
            Prows[pidx][nlvls-o+ii] = Λ
            fbdr.ip[2] *= -1 # undo flip

            # stay
            Prows[pidx][pidx] = max(0., min(1., exp(lpff) - Λ))
        end
    end
    P = collect(hcat(Prows...)')
    return P
end
