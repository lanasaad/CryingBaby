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

function POMDPs.initial_belief(pomdp::CryingBabyPOMDP)
	return SparseCat([HUNGRYₛ, FULLₛ], [0.5, 0.5])
end

function POMDPs.actions(p::CryingBabyPOMDP)
	return [FEEDₐ, IGNOREₐ]
end

function POMDPs.states(p::CryingBabyPOMDP)
    return [HUNGRYₛ, FULLₛ]
end


#function POMDPs.stateindex(p::CryingBabyPOMDP, s::State)
 #   return s == HUNGRYₛ ? 1 : 2
#end


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


#what should be here is beliefupdate=update(oldbelief,observation )

using BeliefUpdaters

#const Belief = Vector{Real};

#updater(pomdp::CryingBabyPOMDP) = DiscreteUpdater(pomdp); #p or pompdp
belief_updater = DiscreteUpdater(p) 
b1=update(belief_updater,b0,a,o) 
