include("helperSim.jl");

par = parLARMEx(seed=1984);

sim = simLARMEx(rng=par.rng);
simData,simData0,SNR = genData(par,sim);

fitData = prepData2Fit(simData,"idL2",["S1","S2"],["E","C"]);
# show(first(fitData,5),allcols=true)

frm = setFormula(fitData);

fit = MixedModels.fit(MixedModel, frm, fitData, progress=false)