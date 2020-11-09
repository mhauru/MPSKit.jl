using MPSKit,MPSKitModels,TensorKit,Test

th = nonsym_xxz_ham()
ts = InfiniteMPS([ℂ^3],[ℂ^48]);
(ts,envs,_) = find_groundstate!(ts,th,Vumps(maxiter=400));
(energies,Bs) = quasiparticle_excitation(th,Float64(pi),ts,envs);
@test energies[1] ≈ 0.41047925 atol=1e-4
