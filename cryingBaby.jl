using POMDPs, POMDPModelTools, Parameters


@with_kw struct CryingBabyPOMDP <: POMDP{State, Action, Observation}
	# Rewards
	r_hungry::Real = -10
	r_feed::Real = -5

	# Transition probability
	p_becomes_hungry::Real = 0.1

	# Observation probabilities
	p_crying_when_hungry::Real = 0.8
	p_crying_when_full::Real = 0.1

	γ = 0.9
end


@enum State HUNGRYₛ FULLₛ
@enum Action FEEDₐ IGNOREₐ
@enum Observation CRYINGₒ QUIETₒ

function POMDPs.transition(pomdp::CryingBabyPOMDP, s::State, a::Action)

	if a == FEEDₐ
		return SparseCat([HUNGRYₛ, FULLₛ], [0, 1])
	elseif s == HUNGRYₛ && a == IGNOREₐ
		return SparseCat([HUNGRYₛ, FULLₛ], [1, 0])
	elseif s == FULLₛ && a == IGNOREₐ
		return SparseCat([HUNGRYₛ, FULLₛ], [pomdp.p_becomes_hungry, 1-pomdp.p_becomes_hungry])
	end
end


function POMDPs.observation(pomdp::CryingBabyPOMDP, s::State, a::Action, sp::State)
	if sp == HUNGRYₛ
		return SparseCat([CRYINGₒ, QUIETₒ],
			             [pomdp.p_crying_when_hungry, 1-pomdp.p_crying_when_hungry])
	else 
		return SparseCat([CRYINGₒ, QUIETₒ],
			             [pomdp.p_crying_when_full, 1-pomdp.p_crying_when_full])
	end
end



function POMDPs.reward(pomdp::CryingBabyPOMDP, s::State, a::Action)
	return (s == HUNGRYₛ ? pomdp.r_hungry : 0) + (a == FEEDₐ ? pomdp.r_feed : 0)
end


function POMDPs.initialstate(pomdp::CryingBabyPOMDP)
	return SparseCat([HUNGRYₛ, FULLₛ], [0.5, 0.5])
end

function POMDPs.initialize_belief(pomdp::CryingBabyPOMDP)
	return SparseCat([HUNGRYₛ, FULLₛ], [0.5, 0.5])
end

function POMDPs.actions(p::CryingBabyPOMDP)
	return [FEEDₐ, IGNOREₐ]
end

p = CryingBabyPOMDP()
s0  = rand(initialstate(p))
b0 = initialize_belief(p)


#define some benchmark policy
#define your policy so that it takes belief as input and output an action


using POMDPPolicies
policy = RandomPolicy(p)

a = action(policy, b0)
sp = rand(transition(p, s0, a))
o=rand(observation(p,s0,a,sp))
r=reward(p,s0,a)

using BeliefUpdaters

up=DiscreteUpdater(p) #update

const Belief = Vector{Real}
update(up, b0)

#function POMDPs.update(::Updater,b::Belief,a::Action,o::Observation)
	#return update((p),b,a,o))

