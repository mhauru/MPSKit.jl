var documenterSearchIndex = {"docs":
[{"location":"man/intro/#Basics-1","page":"Basics","title":"Basics","text":"","category":"section"},{"location":"man/intro/#TensorMap-1","page":"Basics","title":"TensorMap","text":"","category":"section"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"MPSKit works on \"TensorMap\" objects defined in (TensorKit.jl)[https://github.com/Jutho/TensorKit.jl]. These abstract objects can represent not only plain arrays but also symmetric tensors. A TensorMap is a linear map from its domain to its codomain.","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"Initializing a TensorMap can be done using","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"TensorMap(initializer,eltype,codomain,domain);\nTensorMap(inputdat,codomain,domain);","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"As an example, the following creates a random map from ℂ^10 to ℂ^10 (which is equivalent to a random matrix)","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"TensorMap(rand,ComplexF64,ℂ^10,ℂ^10);\ndat = rand(ComplexF64,10,10); TensorMap(dat,ℂ^10,ℂ^10);","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"Similarly, the following creates a symmetric tensor","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"TensorMap(rand,ComplexF64,ℂ[U₁](0=>1)*ℂ[U₁](1//2=>3),ℂ[U₁](1//2=>1,-1//2=>2))","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"TensorKit defines a number of operations on TensorMap objects","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"a = TensorMap(rand,ComplexF64,ℂ^10,ℂ^10);\n3*a; a+a; a*a; a*adjoint(a); a-a; dot(a,a); norm(a);","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"but the primary workhorse is the @tensor macro","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"a = TensorMap(rand,ComplexF64,ℂ^10,ℂ^10);\nb = TensorMap(rand,ComplexF64,ℂ^10,ℂ^10);\n@tensor c[-1;-2]:=a[-1,1]*b[1,-2];","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"creates a new TensorMap c equal to a*b.","category":"page"},{"location":"man/intro/#Creating-states-1","page":"Basics","title":"Creating states","text":"","category":"section"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"Using these TensorMap building blocks we can create states; representing physical objects. An mps tensor is defined as a TensorMap from the bond dimension space (D) to bond dimension space x physical space (D x d). For example, the following creates a finite mps of length 3 :","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"A = TensorMap(rand,ComplexF64,ℂ^1*ℂ^2,ℂ^2);\nB = TensorMap(rand,ComplexF64,ℂ^2*ℂ^2,ℂ^2);\nC = TensorMap(rand,ComplexF64,ℂ^2*ℂ^2,ℂ^1);\nFiniteMPS([A,B,C]);","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"Infinite matrix product states are also supported. A uniform mps representing ... ABABAB... can be created using","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"A = TensorMap(rand,ComplexF64,ℂ^10*ℂ^2,ℂ^10);\nB = TensorMap(rand,ComplexF64,ℂ^10*ℂ^2,ℂ^10);\nMPSCenterGauged([A,B]);","category":"page"},{"location":"man/intro/#Operators-1","page":"Basics","title":"Operators","text":"","category":"section"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"We can act with operators on these states. A number of operators are defined, but the most commonly used one is the MPOHamiltonian. This object represents a regular 1d quantum hamiltonian and can act both on finite and infinite states. As an example, this creates the spin 1 heisenberg :","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"(sx,sy,sz,id) = nonsym_spintensors(1)\n\n@tensor tham[-1 -2;-3 -4]:=sx[-1,-3]*sx[-2,-4]+sy[-1,-3]*sy[-2,-4]+sz[-1,-3]*sz[-2,-4]\nham = MPOHamiltonian(tham)","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"This code is already included in the juliatrack, just call","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"nonsym_xxz_ham();","category":"page"},{"location":"man/intro/#Algorithms-1","page":"Basics","title":"Algorithms","text":"","category":"section"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"Armed with an operator and a state, we can start to do physically useful things; such as finding the groundstate:","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"find_groundstate(state,hamiltonian,algorithm);","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"or perform time evolution:","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"timestep(state,hamiltonian,dt,algorithm);","category":"page"},{"location":"man/intro/#Environments-1","page":"Basics","title":"Environments","text":"","category":"section"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"We can often reuse certain environments in the algorithms, these things are stored in cache objects. The goal is that a user should not have to worry about these objects. Nevertheless, they can be created using:","category":"page"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"params(state,opperator)","category":"page"},{"location":"man/intro/#Tips-and-tricks-1","page":"Basics","title":"Tips & tricks","text":"","category":"section"},{"location":"man/intro/#","page":"Basics","title":"Basics","text":"More information can be found in the documentation, provided someone writes it first.\nThere is an examples folder\nJulia inference is taxed a lot; so use jupyter notebooks instead of re-running a script everytime","category":"page"},{"location":"man/lib/#Library-documentation-1","page":"Library documentation","title":"Library documentation","text":"","category":"section"},{"location":"man/lib/#States-1","page":"Library documentation","title":"States","text":"","category":"section"},{"location":"man/lib/#","page":"Library documentation","title":"Library documentation","text":"FiniteMPS\nFiniteMPO\nMPSCenterGauged\nMPSComoving\nMPSMultiline","category":"page"},{"location":"man/lib/#MPSKit.FiniteMPS","page":"Library documentation","title":"MPSKit.FiniteMPS","text":"FiniteMPS(data::Array)\n\nfinite one dimensional mps\nalgorithms usually assume a right-orthormalized input\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#MPSKit.FiniteMPO","page":"Library documentation","title":"MPSKit.FiniteMPO","text":"FiniteMPO(data::Array)\n\nfinite one dimensional mpo\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#MPSKit.MPSComoving","page":"Library documentation","title":"MPSKit.MPSComoving","text":"MPSComoving(leftstate,window,rightstate)\n\nmuteable window of tensors on top of an infinite chain\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#MPSKit.MPSMultiline","page":"Library documentation","title":"MPSKit.MPSMultiline","text":"2d extension of InfiniteMPS\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#Operators-1","page":"Library documentation","title":"Operators","text":"","category":"section"},{"location":"man/lib/#","page":"Library documentation","title":"Library documentation","text":"MPOHamiltonian\nComAct\nPeriodicMPO","category":"page"},{"location":"man/lib/#MPSKit.MPOHamiltonian","page":"Library documentation","title":"MPSKit.MPOHamiltonian","text":"MPOHamiltonian\n\nrepresents a general periodic quantum hamiltonian\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#MPSKit.ComAct","page":"Library documentation","title":"MPSKit.ComAct","text":"ComAct(ham1,ham2)\n\nActs on an mpo with mpo hamiltonian 'ham1' from below + 'ham2' from above.\nCan therefore represent the (anti) commutator.\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#MPSKit.PeriodicMPO","page":"Library documentation","title":"MPSKit.PeriodicMPO","text":"Represents a periodic (in 2 directions) statmech mpo\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#Environments-1","page":"Library documentation","title":"Environments","text":"","category":"section"},{"location":"man/lib/#","page":"Library documentation","title":"Library documentation","text":"MPSKit.AbstractInfEnv\nMPSKit.PerMPOInfEnv\nMPSKit.MPOHamInfEnv\nMPSKit.FinEnv\nMPSKit.SimpleEnv","category":"page"},{"location":"man/lib/#MPSKit.AbstractInfEnv","page":"Library documentation","title":"MPSKit.AbstractInfEnv","text":"Abstract environment for an infinite state\ndistinct from finite, because we have to recalculate everything when the state changes\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#MPSKit.PerMPOInfEnv","page":"Library documentation","title":"MPSKit.PerMPOInfEnv","text":"This object manages the periodic mpo environments for an MPSMultiline\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#MPSKit.MPOHamInfEnv","page":"Library documentation","title":"MPSKit.MPOHamInfEnv","text":"This object manages the hamiltonian environments for an InfiniteMPS\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#MPSKit.FinEnv","page":"Library documentation","title":"MPSKit.FinEnv","text":"FinEnv keeps track of the environments for FiniteMPS / MPSComoving / FiniteMPO\nIt automatically checks if the queried environment is still correctly cached and if not - recalculates\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#MPSKit.SimpleEnv","page":"Library documentation","title":"MPSKit.SimpleEnv","text":"SimpleEnv does nothing fancy to ensure the correctness of the environments it returns.\nSupports setleftenv! and setrightenv!\nOnly used internally (in idmrg); no public constructor is provided\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#Generic-actions-1","page":"Library documentation","title":"Generic actions","text":"","category":"section"},{"location":"man/lib/#","page":"Library documentation","title":"Library documentation","text":"c_prime\nac_prime\nac2_prime","category":"page"},{"location":"man/lib/#MPSKit.c_prime","page":"Library documentation","title":"MPSKit.c_prime","text":"Zero-site derivative (the C matrix to the right of pos)\n\n\n\n\n\n","category":"function"},{"location":"man/lib/#MPSKit.ac_prime","page":"Library documentation","title":"MPSKit.ac_prime","text":"One-site derivative\n\n\n\n\n\n","category":"function"},{"location":"man/lib/#MPSKit.ac2_prime","page":"Library documentation","title":"MPSKit.ac2_prime","text":"Two-site derivative\n\n\n\n\n\n","category":"function"},{"location":"man/lib/#Groundstate-algorithms-1","page":"Library documentation","title":"Groundstate algorithms","text":"","category":"section"},{"location":"man/lib/#","page":"Library documentation","title":"Library documentation","text":"find_groundstate\nVumps\nIdmrg1\nDmrg\nDmrg2","category":"page"},{"location":"man/lib/#MPSKit.find_groundstate","page":"Library documentation","title":"MPSKit.find_groundstate","text":"find_groundstate(state,ham,alg,pars=params(state,ham))\n\nfind the groundstate for ham using algorithm alg\n\n\n\n\n\n","category":"function"},{"location":"man/lib/#MPSKit.Vumps","page":"Library documentation","title":"MPSKit.Vumps","text":"see https://arxiv.org/abs/1701.07035\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#MPSKit.Idmrg1","page":"Library documentation","title":"MPSKit.Idmrg1","text":"onesite infinite dmrg\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#MPSKit.Dmrg","page":"Library documentation","title":"MPSKit.Dmrg","text":"onesite dmrg\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#MPSKit.Dmrg2","page":"Library documentation","title":"MPSKit.Dmrg2","text":"twosite dmrg\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#Time-evolution-1","page":"Library documentation","title":"Time evolution","text":"","category":"section"},{"location":"man/lib/#","page":"Library documentation","title":"Library documentation","text":"timestep\nTdvp\nTdvp2","category":"page"},{"location":"man/lib/#MPSKit.timestep","page":"Library documentation","title":"MPSKit.timestep","text":"(newstate,newpars) = timestep(state,hamiltonian,dt,alg,pars = params(state,hamiltonian))\n\nevolves state forward by dt using algorithm alg\n\n\n\n\n\n","category":"function"},{"location":"man/lib/#MPSKit.Tdvp","page":"Library documentation","title":"MPSKit.Tdvp","text":"onesite tdvp\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#MPSKit.Tdvp2","page":"Library documentation","title":"MPSKit.Tdvp2","text":"twosite tdvp (works for finite mps's)\n\n\n\n\n\n","category":"type"},{"location":"man/lib/#Bond-dimension-code-1","page":"Library documentation","title":"Bond dimension code","text":"","category":"section"},{"location":"man/lib/#","page":"Library documentation","title":"Library documentation","text":"changebonds\nmanagebonds","category":"page"},{"location":"man/lib/#MPSKit.changebonds","page":"Library documentation","title":"MPSKit.changebonds","text":"Change the bond dimension of state using alg\n\n\n\n\n\n","category":"function"},{"location":"man/lib/#MPSKit.managebonds","page":"Library documentation","title":"MPSKit.managebonds","text":"Manage (grow or shrink) the bond dimsions of state using manager 'alg'\n\n\n\n\n\n","category":"function"},{"location":"man/lib/#Various-1","page":"Library documentation","title":"Various","text":"","category":"section"},{"location":"man/lib/#","page":"Library documentation","title":"Library documentation","text":"dynamicaldmrg\nquasiparticle_excitation","category":"page"},{"location":"man/lib/#MPSKit.dynamicaldmrg","page":"Library documentation","title":"MPSKit.dynamicaldmrg","text":"https://arxiv.org/pdf/cond-mat/0203500.pdf\n\n\n\n\n\n","category":"function"},{"location":"man/lib/#MPSKit.quasiparticle_excitation","page":"Library documentation","title":"MPSKit.quasiparticle_excitation","text":"quasiparticle_excitation calculates the energy of the first excited state at momentum 'moment'\n\n\n\n\n\n","category":"function"},{"location":"#Home-1","page":"Home","title":"Home","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"This code track contains the numerical research and development of the Ghent Quantum Group with regard to tensor network simulation in the julia language. The purpose of this package is to facilitate efficient collaboration between different members of the group.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Topics of research on tensor networks within the realm of this track include:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Tensor network algorithms (excitations, tdvp, vumps, ...)\nMPS routines (MPS diagonalization, Schmidt Decomposition, MPS left and right multiplication, ...)\nThe study of several useful models (nearest neighbour interactions, MPO's, long range interactions, ...)","category":"page"},{"location":"#Table-of-contents-1","page":"Home","title":"Table of contents","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Pages = [\"man/intro.md\",\"man/lib.md\"]\nDepth = 3","category":"page"}]
}
