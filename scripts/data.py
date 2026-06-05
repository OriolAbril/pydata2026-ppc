import warnings

import arviz as az
import numpy as np
import xarray as xr
from xarray_einstats.stats import XrContinuousRV, XrDiscreteRV
from scipy.stats import Normal, Uniform
from scipy import stats

warnings.filterwarnings("ignore", module="zarr")

rng = np.random.default_rng(220715)

# base block
category_coord = (("item",), ["A"]*100+["B"]*50)
obs = XrContinuousRV(Normal, mu=0, sigma=1).rvs(size=150, dims="item", random_state=rng).assign_coords(category=category_coord)
good = XrContinuousRV(Normal, mu=0, sigma=1).rvs(size=(4, 1000, 150), dims=["chain", "draw", "item"], random_state=rng).assign_coords(category=category_coord)
pp_under =  XrContinuousRV(Normal, mu=0, sigma=0.7).rvs(size=(4, 1000, 150), dims=["chain", "draw", "item"], random_state=rng).assign_coords(category=category_coord)
pp_over =  XrContinuousRV(Normal, mu=0, sigma=1.5).rvs(size=(4, 1000, 150), dims=["chain", "draw", "item"], random_state=rng).assign_coords(category=category_coord)
pp_bias =  XrContinuousRV(Normal, mu=0.6, sigma=1).rvs(size=(4, 1000, 150), dims=["chain", "draw", "item"], random_state=rng).assign_coords(category=category_coord)

pp2 = xr.concat(
    (
        XrContinuousRV(Normal, mu=0, sigma=0.9).rvs(size=(4, 1000, 100), dims=["chain", "draw", "item"], random_state=rng),
        XrContinuousRV(Normal, mu=0.4, sigma=1.3).rvs(size=(4, 1000, 50), dims=["chain", "draw", "item"], random_state=rng)
    ),
    dim="item"
).assign_coords(category=category_coord)
pp3 = XrContinuousRV(
    Normal, mu=XrContinuousRV(Normal, mu=obs, sigma=0.2).rvs(), sigma=0.5
).rvs(size=(4, 1000), dims=["chain", "draw"], random_state=rng).assign_coords(category=category_coord)

xr.DataTree.from_dict({"observed_data/y": obs, "posterior_predictive/y": good}).to_zarr("data/base_good.zarr")
xr.DataTree.from_dict({"observed_data/y": obs, "posterior_predictive/y": pp_under}).to_zarr("data/base_underdispersed.zarr")
xr.DataTree.from_dict({"observed_data/y": obs, "posterior_predictive/y": pp_over}).to_zarr("data/base_overdispersed.zarr")
xr.DataTree.from_dict({"observed_data/y": obs, "posterior_predictive/y": pp_bias}).to_zarr("data/base_biased.zarr")

xr.DataTree.from_dict({"observed_data/y": obs, "posterior_predictive/y": pp2}).to_zarr("data/base_ex1.zarr")
xr.DataTree.from_dict({"observed_data/y": obs.sortby(obs), "posterior_predictive/y": pp3.sortby(obs)}).to_zarr("data/base_ex2.zarr")

# discrete block
rng = np.random.default_rng(220715)
ary = stats.geom(0.9).rvs(30*4, random_state=rng)-1
geom = xr.Dataset({"x": (["player"], ary)})
ary = stats.binom(7, 0.2).rvs(30*3, random_state=rng)
binom = xr.Dataset({"x": (["player"], ary)})
defender = xr.concat((geom, binom), dim="player").assign_coords(position=(["player"], ["defender"]*30*7))
defender = defender.isel(player=np.random.default_rng().permutation(len(defender["x"])))
ary = stats.poisson(2.5).rvs(30*6, random_state=rng)
midfield = xr.Dataset({"x": (["player"], ary)}).assign_coords(position=(["player"], ["midfielder"]*30*6))
ary = stats.nbinom(5, 0.4).rvs(30*6, random_state=rng)
attacker = xr.Dataset({"x": (["player"], ary)}).assign_coords(position=(["player"], ["attacker"]*30*6))
ary = stats.bernoulli(0.12).rvs(30, random_state=rng)
goalkeeper = xr.Dataset({"x": (["player"], ary)}).assign_coords(position=(["player"], ["goalkeeper"]*30))
all_players = xr.concat((goalkeeper, defender, midfield, attacker), dim="player")

pp5 = XrDiscreteRV(stats.geom, 0.25).rvs(size=(4, 1000, 600), dims=["chain", "draw", "player"], random_state=rng).assign_coords(position=all_players["position"])-1

rng = np.random.default_rng(220715)
defender = (
    XrDiscreteRV(stats.poisson, 0.7).rvs(size=(4, 1000, 30*7), dims=["chain", "draw", "player"], random_state=rng)
).assign_coords(position=(["player"], ["defender"]*30*7))
midfield = (
    XrDiscreteRV(stats.poisson, 2.5).rvs(size=(4, 1000, 30*6), dims=["chain", "draw", "player"], random_state=rng)
).assign_coords(position=(["player"], ["midfielder"]*30*6))
attacker = (
    XrDiscreteRV(stats.nbinom, 5, 0.4).rvs(size=(4, 1000, 30*6), dims=["chain", "draw", "player"], random_state=rng)
).assign_coords(position=(["player"], ["attacker"]*30*6))
goalkeeper = xr.DataArray(np.zeros((4, 1000, 30), dtype=int), dims=["chain", "draw", "player"]).assign_coords(position=(["player"], ["goalkeeper"]*30))

pp6 = xr.concat((goalkeeper, defender, midfield, attacker), dim="player")

dt5 = xr.DataTree.from_dict({"observed_data/goals": all_players["x"], "posterior_predictive/goals": pp5}).to_zarr("data/discrete_ex1.zarr")
dt6 = xr.DataTree.from_dict({"observed_data/goals": all_players["x"], "posterior_predictive/goals": pp6}).to_zarr("data/discrete_ex2.zarr")

# binary block
rng = np.random.default_rng(4809)

base_probs = XrContinuousRV(stats.beta, 1.5, 2).rvs(size=140, dims=["obs_id"], random_state=rng)
obs = XrDiscreteRV(stats.bernoulli, base_probs).rvs(random_state=rng)
pp1 = XrDiscreteRV(stats.bernoulli, 0.6).rvs(size=(4, 1000, 140), dims=["chain", "draw", "obs_id"], random_state=rng)
pp2 = XrDiscreteRV(stats.bernoulli, base_probs).rvs(size=(4, 1000), dims=["chain", "draw"], random_state=rng)
pp3 = XrDiscreteRV(stats.bernoulli, base_probs/1.6).rvs(size=(4, 1000), dims=["chain", "draw"], random_state=rng)
pp4 = XrDiscreteRV(stats.bernoulli, 1-(1-base_probs)/1.6).rvs(size=(4, 1000), dims=["chain", "draw"], random_state=rng)
pp5 = xr.concat((pp3.sel(obs_id=slice(None, 70)), pp4.sel(obs_id=slice(70, None))), dim="obs_id")
pp6 = XrDiscreteRV(stats.bernoulli, base_probs.mean()).rvs(size=(4, 1000, 140), dims=["chain", "draw", "obs_id"], random_state=rng)

xr.DataTree.from_dict({"observed_data/success": obs, "posterior_predictive/success": pp2}).to_zarr("data/binary-ex1.zarr")
xr.DataTree.from_dict({"observed_data/success": obs, "posterior_predictive/success": pp3}).to_zarr("data/binary-ex2.zarr")
xr.DataTree.from_dict({"observed_data/success": obs, "posterior_predictive/success": pp4}).to_zarr("data/binary-ex3.zarr")
xr.DataTree.from_dict({"observed_data/success": obs, "posterior_predictive/success": pp5}).to_zarr("data/binary-ex4.zarr")
xr.DataTree.from_dict({"observed_data/success": obs, "posterior_predictive/success": pp1}).to_zarr("data/binary-ex5.zarr")
xr.DataTree.from_dict({"observed_data/success": obs, "posterior_predictive/success": pp6}).to_zarr("data/binary-ex6.zarr")


# regression block
rng = np.random.default_rng(952347)

x = XrContinuousRV(Uniform, a=0, b=30).rvs(size=100, dims=["obs_id"], random_state=rng).sortby(lambda v: v)
z = XrContinuousRV(Uniform, a=0, b=10).rvs(size=100, dims=["obs_id"], random_state=rng)
y1 = XrContinuousRV(Normal, mu=1+0.2*x-0.8*z, sigma=2.5).rvs(size=(4, 1000), dims=["chain", "draw"], random_state=rng)
y2 = XrContinuousRV(Normal, mu=1+0.2*x-0.8*z, sigma=3*np.exp(-x/20)).rvs(size=(4, 1000), dims=["chain", "draw"], random_state=rng)

dt1 = xr.DataTree.from_dict(
    {
        "constant_data": {"x": x, "z": z},
        "posterior_predictive/y": y1,
        "observed_data/y": XrContinuousRV(Normal, mu=1+0.2*x-0.8*z, sigma=3/(1+x/10)).rvs(random_state=rng),
    },
    nested=True
)
dt2 = dt1.copy()
dt2["posterior_predictive/y"] = y2

dt1.to_zarr("data/reg-ex1.zarr")
dt2.to_zarr("data/reg-ex2.zarr")

rng = np.random.default_rng(952347)

x = XrContinuousRV(Uniform, a=0, b=30).rvs(size=(5, 70), dims=["group", "obs_id"], random_state=rng)+xr.DataArray(np.linspace(1, 6, 5)**2, dims="group")

intercepts = xr.DataArray(np.linspace(2, 17, 5), dims="group")
sigmas = xr.DataArray(np.linspace(1, 1.7, 5), dims="group")
obs = XrContinuousRV(Normal, mu=-0.2*x+intercepts, sigma=sigmas).rvs(random_state=rng)
                      
y = XrContinuousRV(
    Normal,
    mu=-0.2*x+intercepts,
    sigma=sigmas
).rvs(size=(4, 1000), dims=["chain", "draw"], random_state=rng)
dt3 = xr.DataTree.from_dict(
    {
        "constant_data/x": x,
        "posterior_predictive/y": y,
        "observed_data/y": obs,
    },
).map_over_datasets(lambda ds: ds.assign_coords(group=list("abcde")) if ds else ds)
dt4 = dt3.copy()
sigmas4 = sigmas.copy()
sigmas4[{"group": 1}] = sigmas.isel(group=-1)
sigmas4[{"group": -1}] = sigmas.isel(group=1)
dt4["posterior_predictive/y"] = XrContinuousRV(
    Normal,
    mu=-0.2*x+intercepts,
    sigma=sigmas4
).rvs(size=(4, 1000), dims=["chain", "draw"], random_state=rng).assign_coords(group=list("abcde"))
dt5 = dt3.copy()
dt5["posterior_predictive/y"] = XrContinuousRV(
    Normal,
    mu=-0.15*x+intercepts*0.83,
    sigma=sigmas
).rvs(size=(4, 1000), dims=["chain", "draw"], random_state=rng).assign_coords(group=list("abcde"))

dt3.to_zarr("data/reg-ex3.zarr")
dt4.to_zarr("data/reg-ex4.zarr")
dt5.to_zarr("data/reg-ex5.zarr")
