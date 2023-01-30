# Intraindividual Dynamic Network of Affects
## Linear Autoregressive Mixed-Effects Models for Ecological Momentary Assessment
This repository hosts Jupyter notebooks and Julia code for generating data using exogenous linear autoregressive mixed-effects models (LARMEx) developed as part of [DynaMORE](http://www.dynamore-project.eu) projct.


## Summary
We consider very simple directed intraindividual networks comprising two symptom nodes and one node for external factors. Assuming intensive longitudinal data through **ecological momentary assessment**, we formalize the mathematical representation of such networks by **LARMEx** models. We let every parameter in the model to have fixed and random components aiming at networks that are allowed to have variable structures over reasonable units of time like days or weeks depending on the study design. Then assuming our model is the true data generating process, we simulate data using a predefined set of parameters and investigate the performance and feasibility of this approach in delivering reliable estimates for different choices of the number of observations and the intensity of noise.

## How to use these notebooks?
Assuming that you have downloaded the notebookes:
- install Julia from [julialang.org](https://julialang.org/)
- navigate to the directory of notebooks in `Terminal` and start `Julia`
- the prompt at the command line will change to `julia>` indicating the REPL
- at Julia REPL press `]` to enter package manager (prompt changes to `pkg>`) and execute the following commands
    * `activate .`
    * `instantiate`
- this will install all the required packages including `IJulia` for running the notebooks
- go back to REPL by pressing backspace and run
    * `using IJulia` makes the package available
    * `IJulia.notebook(dir=".")` starts a Jupyter dashboard on your default browser
        - the first time you run `notebook()`, it will prompt you to install Jupyter if it is not found
    * open `index.ipynb` in Jupyter dashboard and proceed
    * you might need to run `using Pkg` and then `Pkg.build("IJulia")` at REPL to complete the setup if you are unable to run the notebook properly
- alternatively you can use REPL without entering the package manager
    * `using Pkg`
    * `Pkg.activate(".")`
    * `Pkg.instantiate()`

It is highly recommended to use a julia environment provided by `Project.toml` and `Manifest.toml` as explained earlier. However, one can proceed from scratch by adding the following packages. Then you might need to modify the code and take care of potential version conflicts.
`Pkg.add(["CSV", "CategoricalArrays", "DataFrames", "Distributions", "IJulia", "Latexify", "LinearAlgebra", "MixedModels", "Printf", "Random"])`

<hr>
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img style="right" alt="Creative Commons Lizenzvertrag" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png"/></img></a>
