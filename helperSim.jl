using LinearAlgebra, Random, Distributions, DataFrames
using MixedModels, DisplayAs, CSV, CategoricalArrays, Latexify

"""
    parLARMEx(;b=[], nVar=2, seed=0, nL2Max=100, nSamp=20000,
               B_ar=.3, B_e=.3, b_var=[.03 .03 2])

Construct the struct holding the settings for generating parameters for simulation.

The default values of the keyword arguments specify a model with two temporally connected
network nodes with one exogenous factor acting on both. 
## keyword arguments:
- `b = []`: if provided with a matrix random-effects are extracted, otherwise generated
- `nVar = 2`: number of temporally connected nodes  
- `seed = 0`: an integer to to replicate, 0 to initialize the random-number-generatoranew
- `nL2Max = 100`: how many random-effects to generate
- `nSamp = 20000`: initial sample size for generating random-effects, 
                   see genRE_CS() for explanation
- `B_ar = .3`: absolute value of fixed-effects autoregressive coefficients,
               `= []` for random values between 0.1 and 0.6
- `B_e = .3`: absolute value of fixed-effects exogenous coefficients, 
              `= []` for random values between 0.1 and 0.6
- `b_var = [.03 .03 .03]`: for constructing a default variance-covariance of RE 
"""
mutable struct parLARMEx
    nVar::Int      
    seed::Int 
    rng ::AbstractRNG  
    nL2Max::Int
    nSamp::Int 
    "Fixed-effects autoregressive coefficients"
    B_AR::Matrix{Float64} 
    "Fixed-effects exogenous coefficients"
    B_E::Vector{Float64}  
    b_var::Matrix{Float64}
    "Variance-covariance of random-effects"
    b_cov::Matrix{Float64}
    "Random-effects autoregressive coefficients of size [`nVar^2` x `nL2Max`]"
    b_ar::Matrix{Float64}
    "Random-effects exogenous coefficients of size [`nVar` x `nL2Max`]"
    b_e::Matrix{Float64}
    "Random-effects constant terms of size [`nVar` x `nL2Max`]"
    b_c::Matrix{Float64}
    function parLARMEx(;b=[], nVar=2, seed=0, nL2Max=100, nSamp=20000,
                        B_ar=.3, B_e=.3, b_var=[.03 .03 .03]) 
        if seed === 0
            rng = MersenneTwister(); 
        else
            rng = MersenneTwister(seed);
        end
        if isempty(B_ar)
            B_AR =  rand(rng,.1:.01:.6,2,2) .* [1 -1 ; -1 1]
        else
            B_AR = B_ar .* [1 -1 ; -1 1] #.6 .* Matrix(1.0I,nVar,nVar) .- .3;
        end
        G = Diagonal(vcat(b_var[1]*ones(nVar^2),repeat(b_var[2:end],inner=nVar)));
        if isempty(b)
            b_ar,b_e,b_c,b_cov = genRE_CS(rng,G,B_AR,nL2Max,nSamp)
        else
            samp = sample(rng,1:size(b)[1],nL2Max)
            b_ar = b[samp,1:nVar^2];
            b_e  = b[samp,(nVar^2+1):(nVar^2+nVar)];
            b_c  = b[samp,(nVar^2+nVar+1):(nVar^2+2*nVar)];
            b_var = b_cov = reshape(Float64[],0,2)  
            nSamp = 0
        end
        if isempty(B_e)
            B_E =  rand(rng,.1:.01:.6,2,1) .* [1 ; -1]
        else
            B_E = B_e .* [1 ; -1]
        end
        return new(nVar,seed,rng,nL2Max,nSamp,B_AR,B_E,b_var,b_cov,b_ar,b_e,b_c)
    end
end

"""
    simLARMEx(;rng=MersenneTwister(), nObsL1=10, nL2=36, sigma=.02, S0_max=.5, E=[])

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
- `S0_max = .5`: determines the amplitude of initial values and exogenous factors, 
                 0.5 to keep trajectories mostly in [-1,1]
- `E = []`: known exogenous factors of size [`nL2` x `nObsL1`], 
            drawn randomly from [0,`S0_max`] by default
"""
mutable struct simLARMEx
    nObsL1::Int       
    nL2::Int       
    sigma::Float64 
    S0::Matrix{Float64}
    E::Vector{Float64}
    function simLARMEx(;rng=MersenneTwister(), nObsL1=10, nL2=36, sigma=.02, S0_max=.5, E=[]) 
        rangeS = range(0,stop=S0_max,length=50)
        if isempty(E)
            E = sample(rng,rangeS,(nL2*nObsL1));
        end
        S0 = sample(rng,[-1 1],nL2) .* hcat(sample(rng,rangeS,nL2),sample(rng,-rangeS,nL2))
        return new(nObsL1,nL2,sigma,S0,E)
    end
end

"""
    genRE_CS(rng,G,B_AR,nL2Max,nSamp)

Generate random-effects parameter.

For consistency it is advised that one instance of a random-number-generator should be 
used throughout the simulation. This function is called inside the constructor of
`parLARMEx` and draws a sample of size `nSamp` from a multivariate normal Distributions
MVN(0,`G`). Then using the fixed-effect autoregressive coefficients, `B_AR`, retains tha 
random-effects for which the eigenvalues of the sum of fixed- and random-effects are less 
tha one. This guarantees that the autoregressive component of the process is stable. 
Finally a number of `nL2Max` parameter sets are returned in the matrices `b_ar`, `b_e`, `b_c`
and their variance-covariance matrix as `b_cov`. 
## Example:
b_ar,b_e,b_c,b_cov = genRE_CS(rng,G,B_AR,nL2Max,nSamp)
"""
function genRE_CS(rng,G,B_AR,nL2Max,nSamp) 
    nVar = size(B_AR)[1] # nE  = length(b_var)-1; d = nVar^2 + nVar*nE;
    mvn = MvNormal(G);
    ri  = rand(rng,mvn,nSamp);
    ind = all.(eachslice(abs.(ri[1:(nVar^2+1*nVar),:]) .< .9,dims=2));
    ri  = ri[:,ind];
    r1  = reshape(ri[1:nVar^2,:],nVar,nVar,size(ri)[2]) .+ B_AR;
    eg  = eigvals.(eachslice(r1,dims=3));
    ind = [!any(abs.(i).>1) for i in eg];
    ri  = ri[:,ind]; 

    a = 1:nVar^2;
    e = (nVar^2+1):(nVar^2+nVar);
    c = (nVar^2+nVar+1):(nVar^2+2*nVar);
    
    samp = sample(rng,axes(ri,2),nL2Max)
    b_ar = ri[a,samp]'
    b_e  = ri[e,samp]'
    b_c  = ri[c,samp]'
    return b_ar,b_e,b_c,cov(ri');
end

"""
    genData(P,S)

Generate simulated data given two `struct` of parameter and data specifications.

It returns the simulated data along with the noiseless data as dataframes and the 
signal-to-noise-ratio `SNR`.
## Example:
simData,simData0,SNR = genData(parLARMEx(),simLARMEx()); 
"""
function genData(P,S)
    data1 = zeros(S.nL2*S.nObsL1,P.nVar)
    data0 = copy(data1)
    rho = 0 # Noise_rho;
    Sigma =  Matrix((S.sigma-rho)I,P.nVar,P.nVar) .+ rho;
    for i in 1:S.nL2 
        BAR = P.B_AR + reshape(P.b_ar[i,:],P.nVar,P.nVar)'; # Julia reshapes columnwise
        E1   = S.E[((i-1)*S.nObsL1+1):(i*S.nObsL1)]
        mvn = MvNormal(Sigma);
        d1 = copy(transpose(rand(P.rng,mvn, S.nObsL1)));
        d0 = 0 .* d1;
        d1[1,:] = S.S0[i,:];
        d0[1,:] = S.S0[i,:];
        ib = 1
        for t in 2:(S.nObsL1)
            d1[t,:] += BAR * d1[t-1,:] + (P.B_E .+ P.b_e[i,:]) * E1[t] + P.b_c[i,:]
            d0[t,:] += BAR * d0[t-1,:] + (P.B_E .+ P.b_e[i,:]) * E1[t] + P.b_c[i,:]
        end
        data1[((i-1)*S.nObsL1+1):(i*S.nObsL1),:] = d1[1:S.nObsL1,:]
        data0[((i-1)*S.nObsL1+1):(i*S.nObsL1),:] = d0[1:S.nObsL1,:]
    end

    tL1 = repeat(1:S.nObsL1,S.nL2);
    idL2 = repeat(1:S.nL2,inner=S.nObsL1);
    if all(P.B_E .== 0)
        simData  = DataFrame(idL2=idL2,tL1=tL1);
        simData0 = DataFrame(idL2=idL2,tL1=tL1);
        cols = ["S1","S2"]
    else
        simData  = DataFrame(idL2=idL2,tL1=tL1,E=S.E);
        simData0 = DataFrame(idL2=idL2,tL1=tL1,E=S.E);
        cols = ["S1","S2","E"]
    end
    for i in 1:P.nVar
        insertcols!(simData ,i+2,cols[i] => data1[:,i]);
        insertcols!(simData0,i+2,cols[i] => data0[:,i]);
    end
    SNR = var(data0) / Sigma[1,1]
    return simData, simData0, SNR; 
end

"""
    prepData2Fit(rawData,idL2,endList,exgList)

Prepare simulated\real data to be fit by LARMEx.

It is assumed that data has been acquired by ecological momentary assessment where 
respondent are observed multiple times in the course of several days or weeks.
## Arguments:
- `rawData`: simulated or real data as a dataframe
- `idL2`: column name for level II units, e.g., an id for each day, `"idL2"` here
- `endList`: list of temporelly connected symptoms, `["S1","S2"]` here
- `exgList`: list of exogenous factors together with the constant terms, `["E","C"]` here 
"""
function prepData2Fit(rawData,idL2,endList,exgList)
    nVar = length(endList)
    nExg = length(exgList)
    colS = map(string,repeat(endList,inner=nVar),repeat(1:nVar,nVar));
    col  = ["idL2";"tL1";"S";colS]
    if "E" in exgList
        col = vcat(col,map(string,repeat(["E"],nVar),1:nVar))
    end
    if "C" in exgList
        col = vcat(col,map(string,repeat(["C"],nVar),1:nVar))
    end
    D = reshape(Float64[],0,3+nVar^2+nVar*nExg)
    l2ID = unique(rawData.idL2);
    for sj in l2ID
        d1 = rawData[rawData.idL2.==sj,:];
        nObs = size(d1,1);
        iD = repeat(Matrix(d1[2:end,1:2]),nVar);
        dS = Matrix(d1[2:nObs,endList]);
        S = reshape(dS,nVar*(nObs-1),1);
        dL = kron(Matrix(1I,nVar,nVar), Matrix(d1[1:(nObs-1),endList]));
        cat = hcat(iD,S,dL)
        if "E" in exgList
            dE = kron(Matrix(1.0I,nVar,nVar),Matrix(d1[2:nObs,["E"]]));
            cat = hcat(cat,dE)
        end
        if "C" in exgList
            C = repeat([1],nObs-1);
            dC = kron(Matrix(1I,nVar,nVar),C);
            cat = hcat(cat,dC);
        end
        D = vcat(D,cat)
    end
    fitData = DataFrame(D,col);
    fitData[!,1:2] = convert.(Int16,fitData[:,1:2]);
    fitData[!,idL2] = categorical(fitData[:,idL2]);
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
11-element Vector{String}: ["idL2","tL1","S","S11","S12","S21","S22","E1","E2","C1","C2"]
"""
function setFormula(fitData) 
    cols = names(fitData);
    idL2 = cols[1];
    colS  = cols[3];
    colRE = cols[4:end];
    indC = findall(x -> occursin("C",x),colRE);
    colFE = copy(colRE);
    deleteat!(colFE,indC);
    frm = string(colS," ~ 0 +",join(colFE,'+')," + (0+",join(colRE,'+'),"|",idL2,")")
    return @eval(@formula($(Meta.parse(frm)))); 
end

"""
    re2csv(P,n,endList,exgList,fName)

Save the random-effects from a `parLARMEx` instance as a CSV file.
"""
function re2csv(P,n,endList,exgList,fName)
    nVar = P.nVar
    col = map(string,repeat(endList,inner=nVar),repeat(1:nVar,nVar));
    D = P.b_ar[1:n,:]
    if "E" in exgList
        col = vcat(col,map(string,repeat(["E"],nVar),1:nVar))
        D = hcat(D,P.b_e[1:n,:])
    end
    if "C" in exgList
        col = vcat(col,map(string,repeat(["C"],nVar),1:nVar))
        D = hcat(D,P.b_c[1:n,:])
    end
    re = DataFrame(D,col)
    CSV.write(fName,re)
end

"""
    fe2csv(P,n,endList,exgList,fName)

Save the fixed-effects from a `parLARMEx` instance as a CSV file.
"""
function fe2csv(P,endList,exgList,fName)
    nVar = P.nVar
    col = map(string,repeat(endList,inner=nVar),repeat(1:nVar,nVar));
    D = reshape(P.B_AR',(1,nVar^2))
    if "E" in exgList
        col = vcat(col,map(string,repeat(["E"],nVar),1:nVar))
        D = hcat(D,reshape(P.B_E,(1,2)))
    end
    if "C" in exgList
        col = vcat(col,map(string,repeat(["C"],nVar),1:nVar))
        D = hcat(D,[0 0])
    end
    if length(D) == length(P.b_var)
        D = vcat(D,reshape(P.b_var,(1,8)))
    else
        bv = hcat(P.b_var[1]*ones(nVar^2),repeat(P.b_var[2:end],inner=nVar))
        D = vcat(D,reshape(bv,(1,8)))
    end
    fe = DataFrame(D,col)
    CSV.write(fName,fe)
end

