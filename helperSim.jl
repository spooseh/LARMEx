using CSV, CategoricalArrays, DataFrames, Distributions
using Latexify, LinearAlgebra, MixedModels, Printf, Random

"""
    parLARMEx(; b=[], nAR=2, seed=0, nL2Max=100, nSamp=20000,
                B_ar=.3, B_e=.3, b_var=[.03 .03 2])

Construct the struct holding the settings for generating parameters for simulation.

The default values of the keyword arguments specify a model with two temporally
connected network nodes with one exogenous factor acting on both. 
## keyword arguments:
- `b = []`: if provided with a matrix random-effects are extracted, 
            otherwise generated
- `nAR = 2`: number of temporally connected nodes  
- `seed = 0`: an integer to to replicate, 0 to initialize the 
              random-number-generatoranew
- `nL2Max = 100`: how many random-effects to generate
- `nSamp = 20000`: initial sample size for generating random-effects, 
                   see genRE_CS() for explanation
- `B_ar = .3`: absolute value of fixed-effects autoregressive coefficients,
               `= []` for random values between 0.1 and 0.6
- `B_e = .3`: absolute value of fixed-effects exogenous coefficients, 
              `= []` for random values between 0.1 and 0.6
- `b_var = [.03 .03 .03]`: for constructing a variance-covariance of RE 
"""
mutable struct parLARMEx
    nAR::Int      
    seed::Int 
    rng ::AbstractRNG  
    nL2Max::Int
    nSamp::Int 
    "Fixed-effects autoregressive coefficients"
    B_AR::Matrix{Float64} 
    "Fixed-effects exogenous coefficients"
    B_E::Matrix{Float64}  
    b_var::Matrix{Float64}
    "Variance-covariance of random-effects"
    b_cov::Matrix{Float64}
    "Random-effects autoregressive coefficients of size [`nAR^2` x `nL2Max`]"
    b_ar::Matrix{Float64}
    "Random-effects exogenous coefficients of size [`nAR` x `nL2Max`]"
    b_e::Matrix{Float64}
    "Random-effects constant terms of size [`nAR` x `nL2Max`]"
    b_c::Matrix{Float64}
    function parLARMEx(;b=[], nAR=2, seed=0, nL2Max=100, nSamp=20000,
                        B_ar=.3, B_e=.3, b_var=[.03 .03 .03])
        if seed === 0
            rng = MersenneTwister(); 
        else
            rng = MersenneTwister(seed);
        end
        nv = div(nAR,2)
        if B_ar isa Number
            B_AR = kron([1 -1 ; -1 1], B_ar .* ones(nv, nv))
            B_AR =  testEig(B_AR)
        elseif size(B_ar)[1] == nAR
            B_AR = B_ar
        else
            B_AR =  kron([1 -1; -1 1], rand(rng, .1:.01:.6, nv, nv))
            B_AR =  testEig(B_AR)
        end
        G = Diagonal(vcat(b_var[1]*ones(nAR^2), repeat(b_var[2:end], inner=nAR)));
        if isempty(b)
            b_ar, b_e, b_c, b_cov = genRE_CS(rng, G, B_AR, nL2Max, nSamp)
        else
            samp = sample(rng, 1:size(b)[1], nL2Max)
            b_ar = b[samp, 1:nAR^2];
            b_e  = b[samp, (nAR^2+1):(nAR^2+nAR)];
            b_c  = b[samp, (nAR^2+nAR+1):(nAR^2+2*nAR)];
            b_var = b_cov = reshape(Float64[], 0, 2)  
            nSamp = 0
        end
        if isempty(B_e)
            B_E =  kron([1; -1], rand(rng, .1:.01:.6, nv, 1)) 
        else
            B_E = kron([1; -1], B_e .* ones(nv,1))
        end
        return new(nAR, seed, rng, nL2Max, nSamp, B_AR, B_E, b_var, b_cov, b_ar, b_e, b_c)
    end
end

"""
    testEig(x)

Test if any of the eigenvalues of x is greater than 1.

If any of the eigenvalues of the matrix of autoregressive coefficients is greater than on,
some of the off-diagonal elements are randomly changed to zero until all the eigenvalues
are less than one.
"""
function testEig(x)
    n = size(x)[1]
    y = copy(x)
    while any(abs.(eigvals(x)) .> 1)
        x = y .* (rand(n,n) .> .2)
        x[diagind(x)] .= .3
    end
    return x
end

"""
    simLARMEx(;rng=MersenneTwister(), nObsL1=10, nL2=36, sigma=.02, M0_max=.5, E=[])

Construct the `struct` holding the settings for generating simulated data.

By the default values of the keyword arguments it is assumed that the data generating 
process has a two-level structure. The multiple observations are made on level I and 
these are nested in level II units.
## keyword arguments:
- `rng = MersenneTwister()`: feed `parLARMEx.rng` for consistency and replication, 
                             initialized anew by default
- `nObsL1 = 10`: number of observation on level I 
- `nL2 = 36`: number of units on level II
- `sigma = .02`: variance of noise 
- `M0_max = .5`: determines the amplitude of initial values and exogenous factors, 
                 0.5 to keep trajectories mostly in [-1,1]
- `E = []`: known exogenous factors of size [`nL2` x `nObsL1`], 
            drawn randomly from [0,`M0_max`] by default
"""
mutable struct simLARMEx
    nAR::Int      
    nObsL1::Int       
    nL2::Int       
    sigma::Float64 
    M0::Matrix{Float64}
    E::Vector{Float64}
    function simLARMEx(; rng=MersenneTwister(), nAR=2, nObsL1=10, nL2=36, sigma=.02, M0_max=.5, E=[]) 
        rangeM = range(0, stop=M0_max, length=50);
        if E isa Number
            E = E * ones(nL2*nObsL1)
        elseif length(E) == (nL2*nObsL1)
            E = E
        else 
            E = sample(rng, rangeM, (nL2*nObsL1));
        end
        nv = div(nAR, 2)
        M0 = sample(rng,[-1 1], nL2) .* hcat(sample(rng, rangeM, (nL2,nv)), sample(rng, -rangeM, (nL2, nv)))
        return new(nAR, nObsL1, nL2, sigma, M0, E)
    end
end

"""
    genRE_CS(rng, G, B_AR, nL2Max, nSamp)

Generate random-effects parameter.

For consistency it is advised that one instance of a random-number-generator should be 
used throughout the simulation. This function is called inside the constructor of
`parLARMEx` and draws a sample of size `nSamp` from a multivariate normal Distributions
MVN(0, `G`). Then using the fixed-effect autoregressive coefficients, `B_AR`, retains tha 
random-effects for which the eigenvalues of the sum of fixed- and random-effects are less 
tha one. This guarantees that the autoregressive component of the process is stable. 
Finally a number of `nL2Max` parameter sets are returned in the matrices `b_ar`, `b_e`, `b_c`
and their variance-covariance matrix as `b_cov`. 
## Example:
b_ar, b_e, b_c, b_cov = genRE_CS(rng, G, B_AR, nL2Max, nSamp)
"""
function genRE_CS(rng, G, B_AR, nL2Max, nSamp) 
    nAR = size(B_AR)[1] # nE  = length(b_var)-1; d = nAR^2 + nAR*nE;
    mvn = MvNormal(G);
    ri  = rand(rng, mvn, nSamp);
    ind = all.(eachslice(abs.(ri[1:(nAR^2+1*nAR), :]) .< .9, dims=2));
    ri  = ri[:,ind];
    r1  = reshape(ri[1:nAR^2, :], nAR, nAR, size(ri)[2]) .+ B_AR;
    eg  = eigvals.(eachslice(r1, dims=3));
    ind = [!any(abs.(i).>1) for i in eg];
    ri  = ri[:, ind]; 

    a = 1:nAR^2;
    e = (nAR^2+1):(nAR^2+nAR);
    c = (nAR^2+nAR+1):(nAR^2+2*nAR);
    
    samp = sample(rng, axes(ri, 2), nL2Max)
    b_ar = ri[a, samp]'
    b_e  = ri[e, samp]'
    b_c  = ri[c, samp]'
    return b_ar, b_e, b_c, cov(ri');
end

"""
    genData(P, S)

Generate simulated data given two `struct` of parameter and data specifications.

It returns the simulated data along with the noiseless data as dataframes and the 
signal-to-noise-ratio `SNR`.
## Example:
dfData,dfData0,SNR = genData(parLARMEx(), simLARMEx()); 
"""
function genData(P, S)
    data1 = zeros(S.nL2*S.nObsL1, P.nAR)
    data0 = copy(data1)
    rho = 0 # Noise_rho;
    Sigma =  Matrix((S.sigma-rho)I, P.nAR, P.nAR) .+ rho
    for i in 1:S.nL2 
        BAR = P.B_AR + reshape(P.b_ar[i,:], P.nAR, P.nAR)'; # Julia reshapes columnwise
        E1   = S.E[((i-1)*S.nObsL1+1):(i*S.nObsL1)]
        mvn = MvNormal(Sigma);
        d1 = copy(transpose(rand(P.rng, mvn, S.nObsL1)))
        d0 = 0 .* d1;
        d1[1,:] = S.M0[i,:]
        d0[1,:] = S.M0[i,:]
        ib = 1
        for t in 2:(S.nObsL1)
            d1[t,:] += BAR * d1[t-1,:] + (P.B_E .+ P.b_e[i,:]) * E1[t] + P.b_c[i,:]
            d0[t,:] += BAR * d0[t-1,:] + (P.B_E .+ P.b_e[i,:]) * E1[t] + P.b_c[i,:]
        end
        data1[((i-1)*S.nObsL1+1):(i*S.nObsL1),:] = d1[1:S.nObsL1,:]
        data0[((i-1)*S.nObsL1+1):(i*S.nObsL1),:] = d0[1:S.nObsL1,:]
    end

    tL1 = repeat(1:S.nObsL1, S.nL2);
    idL2 = repeat(1:S.nL2, inner=S.nObsL1);
    colM = map(string, repeat(["M"], P.nAR), 1:P.nAR)
    if all(S.E .== 0)
        dfData  = DataFrame(idL2=idL2, tL1=tL1);
        dfData0 = DataFrame(idL2=idL2, tL1=tL1);
        cols = colM
    else
        dfData  = DataFrame(idL2=idL2, tL1=tL1, E=S.E);
        dfData0 = DataFrame(idL2=idL2, tL1=tL1, E=S.E);
        cols = vcat(colM, ["E"])
    end
    for i in 1:P.nAR
        insertcols!(dfData , i+2, cols[i] => data1[:,i]);
        insertcols!(dfData0, i+2, cols[i] => data0[:,i]);
    end
    SNR = var(data0) / Sigma[1,1]
    return dfData, dfData0, SNR; 
end

"""
    prepData2Fit(rawData, idL2, arList, exList)

Prepare simulated\real data to be fit by LARMEx.

It is assumed that data has been acquired by ecological momentary assessment where 
respondent are observed multiple times in the course of several days or weeks.
## Arguments:
- `rawData`: simulated or real data as a dataframe
- `idL2`: column name for level II units, e.g., an id for each day, `"idL2"` here
- `arList`: list of temporelly connected symptoms, `["M1","M2"]` here
- `exList`: list of exogenous factors together with the constant terms, `["E","C"]` here 
"""
function prepData2Fit(rawData, idL2, arList, exList)
    nAR = length(arList)
    nEX = length(exList)
    colM = map(string, repeat(arList, inner=nAR), repeat(1:nAR, nAR));
    col  = ["idL2"; "tL1"; "M"; colM]
    if "E" in exList
        col = vcat(col, map(string, repeat(["E"], nAR), 1:nAR))
    end
    if "C" in exList
        col = vcat(col, map(string, repeat(["C"], nAR), 1:nAR))
    end
    D = reshape(Float64[], 0, 3+nAR^2+nAR*nEX)
    l2ID = unique(rawData.idL2);
    for sj in l2ID
        d1 = rawData[rawData.idL2.==sj, :];
        nObs = size(d1,1);
        iD = repeat(Matrix(d1[2:end, 1:2]), nAR);
        dS = Matrix(d1[2:nObs, arList]);
        S = reshape(dS, nAR*(nObs-1), 1);
        dL = kron(Matrix(1I, nAR, nAR), Matrix(d1[1:(nObs-1), arList]));
        cat = hcat(iD, S, dL)
        if "E" in exList
            dE = kron(Matrix(1.0I, nAR, nAR), Matrix(d1[2:nObs, ["E"]]));
            cat = hcat(cat, dE)
        end
        if "C" in exList
            C = repeat([1], nObs-1);
            dC = kron(Matrix(1I, nAR, nAR), C);
            cat = hcat(cat, dC);
        end
        D = vcat(D, cat)
    end
    fitData = DataFrame(D, col);
    fitData[!, 1:2] = convert.(Int16, fitData[:, 1:2]);
    fitData[!, idL2] = categorical(fitData[:, idL2]);
    return fitData; 
end

"""
    setFormula(fitData)

Generate a mixed-effects formula from a specifically formatted dataframe.

The input is supposed to have a special format where in a two-level longitudinal data 
the level II identifiers are in the first and the stacked observation in the third columns.
the columns `4:end` are supposed to represent the network structure and constant terms.
## Example:
`names(fitData)`:
11-element Vector{String}: ["idL2","tL1","M","M11","M12","M21","M22","E1","E2","C1","C2"]
"""
function setFormula(fitData) 
    cols = names(fitData);
    idL2 = cols[1];
    colM  = cols[3];
    colRE = cols[4:end];
    indC = findall(x -> occursin("C", x), colRE);
    colFE = copy(colRE);
    deleteat!(colFE, indC);
    frm = string(colM, " ~ 0 +", join(colFE, '+'), " + (0+", join(colRE, '+'), "|", idL2, ")")
    return @eval(@formula($(Meta.parse(frm)))); 
end

"""
    re2csv(P, n, arList, exList, fName)

Save the random-effects from a `parLARMEx` instance as a CSV file.
"""
function re2csv(P, n, arList, exList, fName)
    nAR = P.nAR
    col = map(string, repeat(arList, inner=nAR), repeat(1:nAR, nAR));
    D = P.b_ar[1:n, :]
    if "E" in exList
        col = vcat(col, map(string, repeat(["E"], nAR), 1:nAR))
        D = hcat(D, P.b_e[1:n, :])
    end
    if "C" in exList
        col = vcat(col, map(string, repeat(["C"], nAR), 1:nAR))
        D = hcat(D, P.b_c[1:n, :])
    end
    re = DataFrame(D, col)
    CSV.write(fName, re)
end

"""
    fe2csv(P, arList, exList, fName)

Save the fixed-effects from a `parLARMEx` instance as a CSV file.
"""
function fe2csv(P, arList, exList, fName)
    nAR = P.nAR
    col = map(string, repeat(arList, inner=nAR), repeat(1:nAR, nAR));
    D = reshape(P.B_AR', (1, nAR^2))
    if "E" in exList
        col = vcat(col, map(string, repeat(["E"], nAR), 1:nAR))
        D = hcat(D, reshape(P.B_E, (1, nAR)))
    end
    if "C" in exList
        col = vcat(col, map(string, repeat(["C"], nAR), 1:nAR))
        D = hcat(D, zeros(1,nAR))
    end
    if length(D) == length(P.b_var) # full b_var is provided
        D = vcat(D, reshape(P.b_var, (1, nAR^2+2*nAR)))
    else
        bv = hcat(P.b_var[1]*ones(nAR^2), repeat(P.b_var[2:end], inner=nAR))
        D = vcat(D, reshape(bv, (1, nAR^2+2*nAR)))
    end
    fe = DataFrame(D, col)
    CSV.write(fName, fe)
end

"""
    loopSim(csvDir, M0_max, sigma, n, P)

Simulate data given the same fixed-effects parameters and different reando effects n times.

## Arguments
- `csvDir`: results are saved in this path
- `M0_max`: amplitude of initial values
- `sigma`: standard deviation of noise
- `n`: number of Bootstrap simulations
- `P`: model parameters to simulate data
"""
function loopSim(csvDir, M0_max, sigma, n, P)
    nv = P.nAR
    arList = map(string, repeat(["M"], nv), 1:nv);
    exList = ["E", "C"];
    feh = zeros(nSim, nv^2+nv)
    sig = zeros(nSim, nv^2+2*nv+1)
    fehName = joinpath(csvDir, @sprintf("feh_n%02d.csv", n))
    sigName = joinpath(csvDir, @sprintf("sig_n%02d.csv", n))
    nr = 0
    if all(isfile.([fehName, sigName]))
        fehD = CSV.read(fehName, DataFrame)
        sigD = CSV.read(sigName, DataFrame)
        nr = min(size(fehD)[1], size(sigD)[1])
        feh[1:nr,:] .= fehD[1:nr,:]
        sig[1:nr,:] .= sigD[1:nr,:]
    end
    for j in 1:nSim
        par = parLARMEx(nAR=P.nAR, B_ar=P.B_AR, b=hcat(P.b_ar,P.b_e,P.b_c), nL2Max=n);
        sim = simLARMEx(nAR=P.nAR, nL2=n, M0_max=M0_max, sigma=sigma);
        simData,_,_ = genData(par, sim);
        
        reName  = joinpath(csvDir, "re",  @sprintf("re_n%02d_%03d.csv",n,j))
        rehName = joinpath(csvDir, "reh", @sprintf("reh_n%02d_%03d.csv",n,j))
        if all(isfile.([reName, rehName])) & (j <= nr)
            continue
        end
        @printf("\r s: %.2f, n: %02d, sim: %03d\n", sigma, n, j);
        fitData = prepData2Fit(simData, "idL2", arList, exList);
        frm = setFormula(fitData);
        res = MixedModels.fit(MixedModel, frm, fitData, progress=false);
        
        re2csv(par, n, arList, exList, reName)
        CSV.write(rehName, DataFrame(only(raneftables(res))))
        
        feh[j,:] = coef(res)
        col = coefnames(res)
        CSV.write(fehName, DataFrame(feh[1:j,:], col))
        
        sig[j,:] = vcat(collect(res.sigmas.idL2), res.sigma)
        col = vcat(col, map(string, repeat(["C"], nv), 1:nv), ["sigma"])
        CSV.write(sigName, DataFrame(sig[1:j,:], col))
    end
end