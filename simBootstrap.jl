# using Distributed # for multithread loops
include("helperSim.jl")

nAR = 2
baseDir = @sprintf("%s/LARMExMood/Bootstrap/bs_V%d_bVar03_demo", homedir(), nAR)
if !isdir(baseDir)
    mkpath(baseDir)
end
nSim = 2
b_var=.03 * [1 1 1]; M0_max=.5; sigma=.02;
nDay = 34:2:37
P = parLARMEx(nAR=nAR, seed=1984);
for n in nDay
# Threads.@threads for n in nDay 
    csvDir = joinpath(baseDir, @sprintf("sig%.2f/n%02d/", sigma, n)) 
    if !isdir(csvDir)
        mkpath(csvDir)
    end
    dirs = ["re" "reh"]
    for d in dirs
        cD = joinpath(csvDir, d)
        if !isdir(cD)
            mkpath(cD)
        end
    end
    try
        loopSim(csvDir, M0_max, sigma, n, P)
    catch e
        bt = catch_backtrace()
        msg = sprint(showerror, e, bt)
        println(msg)
        break
    end
end

    