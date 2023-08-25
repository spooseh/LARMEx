include("helperSim.jl");

seed = 1984
# seed = 0
if seed === 0
    rng = MersenneTwister(); 
else
    rng = MersenneTwister(seed);
end
par = parLARMEx(nAR=2, rng=rng);

sigma = .02
sim = simLARMEx(nAR=par.nAR, rng=par.rng, sigma=sigma);
simData, simData0, SNR = genData(par, sim);

arList = map(string, repeat(["M"], par.nAR), 1:par.nAR)
exList = ["E", "C"]
fitData = prepData2Fit(simData, rng, "idL2", arList, exList); # No missing
# fitData = prepData2Fit(simData, rng, "idL2", arList, exList, miss=.2); # 20% missing
# show(first(fitData, 5), allcols=true)

frm = setFormula(fitData);

fit = MixedModels.fit(MixedModel, frm, fitData, REML=true, progress=true)