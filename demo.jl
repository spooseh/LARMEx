include("helperSim.jl");

seed = 1984
seed = 0
par = parLARMEx(nAR=2, seed=seed);

sigma = .02
sim = simLARMEx(nAR=par.nAR, rng=par.rng, sigma=sigma);
simData, simData0, SNR = genData(par, sim);

arList = map(string, repeat(["M"], par.nAR), 1:par.nAR)
exList = ["E", "C"]
fitData = prepData2Fit(simData, "idL2", arList, exList);
# show(first(fitData, 5), allcols=true)

frm = setFormula(fitData);

fit = MixedModels.fit(MixedModel, frm, fitData, progress=true)