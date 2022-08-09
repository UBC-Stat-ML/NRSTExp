# Loading the data and instantiating the model
function HierarchicalModel()
    Y     = readdlm(pkgdir(NRSTExp, "data", "simulated8schools.csv"), ',', Float64)
    model = _HierarchicalModel(Y)
    return TuringTemperedModel(model)
end
