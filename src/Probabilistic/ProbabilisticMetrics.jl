module ProbabilisticMetrics
using ..BasicStructs, Parameters
export calculateCohenCappa


"""
calculate Cohen Cappa  based on precalulated constants
TnTpFpFns - list of basic metrics - in the order [tn,tp,fp,fn ] 
return Cohen Cappa
"""
function calculateCohenCappa(tn, tp, fp, fn)::Float64
	agreement = tp + tn
	chance_0 = (tn + fn) * (tn + fp)
	chance_1 = (fp + tp) * (fn + tp)
	chance = chance_0 + chance_1
	sum = (tn + fn + fp + tp)
	chance::Float32 = chance / sum
	return (agreement - chance) / (sum - chance)
end #calculateVolumeMetric

end#ProbabilisticMetrics