using NRSTExp

dfres = dispatch("ess_versus_cost", "MvNormal", 0.99)

# # PROBLEM: gadfly does not support linestyle key, see https://github.com/GiovineItalia/Gadfly.jl/issues/1558#issuecomment-968260747
# using Gadfly
# using DataFrames

# dfplot = rename(vcat(
#     insertcols!(rename(dfres[:, Not(:xp)], :xs => :cost), :comp => "Serial"),
#     insertcols!(rename(dfres[:, Not(:xs)], :xp => :cost), :comp => "Parallel")
# ), :y => :ess)
# plot(
#     dfplot,
#     x=:cost, y=:ess, color=:model, linestyle=:comp,
#     layer(group=:comp, Stat.smooth(method=:loess), Geom.line),
#     Scale.linestyle_discrete(levels=["Serial","Parallel"]),
#     # 
#     # Scale.linestyle_discrete()
#     # Guide.xlabel("SepalLength"), Guide.ylabel("SepalWidth"),
#     # Guide.colorkey(title="Species")
# )