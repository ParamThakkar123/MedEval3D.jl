"""
holding kernel and necessery functions to calclulate number of true positives,
true negatives, false positives and negatives par image and per slice
using synergism described by Taha et al. this will enable later fast calculations of many other metrics
"""
module TpfpfnKernel
export getTpfpfnData, prepareForconfusionTableMetricsNoSliceWise, @iterateLinearlyForTPTF, addToTp, addToFp, addToFn

using CUDA, ..ReductionUtils, ..CUDAGpuUtils, ..IterationUtils, ..MemoryUtils, ..CUDAAtomicUtils, StaticArrays
using ..MainOverlap, ..RandIndex, ..ProbabilisticMetrics, ..VolumeMetric, ..InformationTheorhetic


"""
prepares all needed data structures and run occupancy API to enable running occupancy API to get the optimal number of blocks and threads per block
goldGPU , segmGPU - example of arrays of gold standard and algorithm output they need to be of the same dimensions 
numberToLooFor - number we will look for in the arrays
conf - configuration struct telling which metrics exactly we want
"""
function prepareForconfusionTableMetricsNoSliceWise(conf)
	tp, tn, fp, fn = CUDA.zeros(UInt32, 1), CUDA.zeros(UInt32, 1), CUDA.zeros(UInt32, 1), CUDA.zeros(UInt32, 1)
	mainArrDims = (2, 2, 2)
	numberToLooFor = 1
	# sliceMetricsTupl= (CUDA.zeros(mainArrDims[3]),CUDA.zeros(mainArrDims[3]),CUDA.zeros(mainArrDims[3]),CUDA.zeros(mainArrDims[3]),CUDA.zeros(mainArrDims[3]),CUDA.zeros(mainArrDims[3]),CUDA.zeros(mainArrDims[3]) ,CUDA.zeros(mainArrDims[3]),CUDA.zeros(mainArrDims[3]),CUDA.zeros(mainArrDims[3]),CUDA.zeros(mainArrDims[3]) )
	metricsTuplGlobal = zeros(Float64, 11) #  (CUDA.zeros(1),CUDA.zeros(1),CUDA.zeros(1),CUDA.zeros(1)
	#,CUDA.zeros(1),CUDA.zeros(1),CUDA.zeros(1)
	#,CUDA.zeros(1),CUDA.zeros(1),CUDA.zeros(1),CUDA.zeros(1) )#eleven entries
	totalNumbOfVoxels = (mainArrDims[1] * mainArrDims[2] * mainArrDims[3])
	pixPerSlice = mainArrDims[1] * mainArrDims[2]
	iterLoop = 5
	args = (#sliceMetricsTupl,
		tp, tn, fp, fn, mainArrDims, totalNumbOfVoxels, iterLoop, pixPerSlice, numberToLooFor#numberToLooFor
		# ,metricsTuplGlobal
		, conf)

	get_shmem(threads) = 4 * 33

	threads, blocks = getThreadsAndBlocksNumbForKernel(get_shmem, getBlockTpFpFn, (CUDA.zeros(2, 2, 2), CUDA.zeros(2, 2, 2), args...))
	#corrections for loop x,y,z variables
	pixPerSlice = cld(totalNumbOfVoxels, blocks)
	iterLoop = UInt32(fld(pixPerSlice, threads[1] * threads[2]))
	args = (#sliceMetricsTupl,
		tp, tn, fp, fn, mainArrDims, totalNumbOfVoxels, iterLoop, pixPerSlice, numberToLooFor#numberToLooFor
		# ,metricsTuplGlobal
		, conf)
	return (args, threads, blocks, metricsTuplGlobal)
end

"""
returning the data  from a kernel that  calclulate number of true positives,
true negatives, false positives and negatives par image and per slice in given data 
goldBoolGPU - array holding data of gold standard bollean array
segmBoolGPU - boolean array with the data we want to compare with gold standard
tp,tn,fp,fn - holding single values for true positive, true negative, false positive and false negative
sliceMetricsTupl - tuple holding slicewiseMetrics
metricsTuplGlobal - tuple holding required metrics of all image 
threadNumPerBlock = threadNumber per block defoult is 512
numberToLooFor - num
conf- adapted ConfigurtationStruct - used to pass information what metrics should be

"""
function getTpfpfnData!(goldGPU, segmGPU, args, threads, blocks, metricsTuplGlobal, numberToLooFor, conf)

	for i in 1:4
		CUDA.fill!(args[i], 0)
	end

	for i in 1:length(metricsTuplGlobal)
		metricsTuplGlobal[i] = 0
		#CUDA.fill!(args[11][i],0)
	end
	mainArrDims = size(goldGPU)
	totalNumbOfVoxels = (mainArrDims[1] * mainArrDims[2] * mainArrDims[3])
	pixPerSlice = cld(totalNumbOfVoxels, blocks)
	iterLoop = UInt32(fld(pixPerSlice, threads[1] * threads[2]))

	args = (#sliceMetricsTupl,
		args[1], args[2], args[3], args[4], mainArrDims, totalNumbOfVoxels, iterLoop, pixPerSlice, numberToLooFor#numberToLooFor
		# ,metricsTuplGlobal
		, conf)

	#get tp,fp,fna and slicewise results if required
	@cuda threads = threads blocks = blocks getBlockTpFpFn(vec(goldGPU), vec(segmGPU), args...) #args[8][3]  is number of slices ...
	getMetricsCPU(args[1][1], args[3][1], args[4][1], (args[6] - (args[1][1] + args[3][1] + args[4][1])), metricsTuplGlobal, args[10], 1)
	return args
end#getTpfpfnData

"""
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
goldBoolGPU - array holding data of gold standard bollean array
segmBoolGPU - boolean array with the data we want to compare with gold standard
tp,tn,fp,fn - holding single values for true positive, true negative, false positive and false negative
sliceMetricsTupl - tuple of arrays holding slice wise results for tp,fp,fn and all metrics of intrest - in case we would not be intrested in some metric tuple in this spot will have the array of length 1
loopNumb - number of times the single lane needs to loop in order to get all needed data
sliceEdgeLength - length of edge of the slice we need to square this number to get number of pixels in a slice
conf- adapted ConfigurtationStruct - used to pass information what metrics should be
"""
function getBlockTpFpFn(goldGPU, segmGPU#segmBoolGPU
	#,sliceMetricsTupl
	, tp, tn, fp, fn, arrDims, totalNumbOfVoxels, iterLoop, pixPerSlice, numberToLooFor#numberToLooFor
	#,metricsTuplGlobal
	, conf)

	shmemSum = @cuStaticSharedMem(UInt32, (33, 3))
	shmemblockData = @cuStaticSharedMem(UInt32, (32, 32, 3))
	boolGold = false
	boolSegm = false

	@iterateLinearlyMultipleBlocks(iterLoop, pixPerSlice, totalNumbOfVoxels,
		#inner expression
		begin
			# CUDA.@cuprint "i $(i)  val $(goldGPU[i])"
			#updating variables needed to calculate means

			boolGold = goldGPU[i] == numberToLooFor
			boolSegm = segmGPU[i] == numberToLooFor

			@inbounds shmemblockData[threadIdxX(), threadIdxY(), (boolGold&boolSegm+boolSegm+1)] += (boolGold | boolSegm)
		end)

	# tell what variables are to be reduced and by what operation
	@redWitAct(offsetIter, shmemSum, shmemblockData[threadIdxX(), threadIdxY(), 1], +, shmemblockData[threadIdxX(), threadIdxY(), 2], +, shmemblockData[threadIdxX(), threadIdxY(), 3], +)
	sync_threads()
	@addAtomic(shmemSum, fn, fp, tp)
	return
end

"""
this will be invoked in order to get global metrics besed on tp,fp,fn calculated in previous kernel 
tp,tn,fp - true positive, true negative, false positive
totalNumbOfVoxels - number of voxels in all image
metricsTuplGlobal - tuple with array of length one for storing global metrics
conf - ConfigurtationStruct - marking in what metrics we are intrested in 
"""
function getGlobalMetricsKernel(tp, fp, fn, totalNumbOfVoxels::Int64, metricsTuplGlobal, conf)
	getMetrics(tp[1], fp[1], fn[1], (totalNumbOfVoxels - (tp[1] + fn[1] + fp[1])), metricsTuplGlobal, conf, 1)
	return
end


"""
increments given UINT16 given both boolGold and boolSegm are true
"""
function addToTp(boolGold::Bool, boolSegm::Bool, tp::UInt16)
	Base.llvmcall("""
	%4 = and i8 %0, %1
	%5 = zext i8 %4 to i16
	%6 = add i16 %2,%5
	ret i16 %6""", UInt16, Tuple{Bool, Bool, UInt16}, boolGold, boolSegm, tp)
end

"""
increments given UINT16 when boolGold is false and boolSegm is true
"""
function addToFp(boolGold::Bool, boolSegm::Bool, tp::UInt16)
	Base.llvmcall("""
	%4 = xor i8 %0, %1
	%5 = and i8 %4, %1
	%6 = zext i8 %5 to i16
	%7 = add i16 %2,%6
	ret i16 %7""", UInt16, Tuple{Bool, Bool, UInt16}, boolGold, boolSegm, tp)
end

"""
increments given UINT16 when boolGold is true and boolSegm is false
"""
function addToFn(boolGold::Bool, boolSegm::Bool, tp::UInt16)
	Base.llvmcall("""
	%4 = xor i8 %0, %1
	%5 = and i8 %4, %1
	%6 = zext i8 %5 to i16
	%7 = add i16 %2,%6
	ret i16 %7""", UInt16, Tuple{Bool, Bool, UInt16}, boolGold, boolSegm, tp)
end

"""
loading data into results
sliceMetricsTupl
   1) true positives
   2) false positives
   3) flse negatives
   4) dice
   5) jaccard
   6) gce
   7) randInd
   8) cohen kappa
   9) volume metric
   10) mutual information
   11) variation of information
positionToUpdate - index at which we want to update the metric - in case of slice wise metrics it will be number of slice = block idx

"""
function getMetrics(tp, fp, fn, tn, sliceMetricsTupl, conf, positionToUpdate)
	# @ifXY 1 7 if (conf.dice ) @inbounds sliceMetricsTupl[4][positionToUpdate]=   MainOverlap.dice(tp,fp, fn) end 
	# @ifXY 1 8  if (conf.jaccard ) @inbounds sliceMetricsTupl[5][positionToUpdate]= MainOverlap.jaccard(tp,fp, fn) end 
	# @ifXY 1 9  if (conf.gce ) @inbounds sliceMetricsTupl[6][positionToUpdate]= MainOverlap.gce(tn,tp,fp, fn) end 
	# @ifXY 1 10 if (conf.randInd ) @inbounds sliceMetricsTupl[7][positionToUpdate]=  RandIndex.calculateAdjustedRandIndex(tn,tp,fp, fn) end 
	# @ifXY 1 11 if (conf.kc ) @inbounds sliceMetricsTupl[8][positionToUpdate]=  ProbabilisticMetrics.calculateCohenCappa(tn,tp,fp, fn  ) end 
	# @ifXY 1 12  if (conf.vol ) @inbounds sliceMetricsTupl[9][positionToUpdate]= VolumeMetric.getVolumMetric(tp,fp, fn ) end 
	# @ifXY 1 13 if (conf.mi ) @inbounds sliceMetricsTupl[10][positionToUpdate]=   InformationTheorhetic.mutualInformationMetr(tn,tp,fp, fn) end 
	# @ifXY 1 14 if (conf.vi ) @inbounds sliceMetricsTupl[11][positionToUpdate]=  InformationTheorhetic.variationOfInformation(tn,tp,fp, fn) end 
end

function getMetricsCPU(tp, fp, fn, tn, sliceMetricsTupl, conf, positionToUpdate)
	tnPrim = tn
	if (conf.dice)
		@inbounds sliceMetricsTupl[4] = MainOverlap.dice(tp, fp, fn)
	end

	if (conf.jaccard)
		@inbounds sliceMetricsTupl[5] = MainOverlap.jaccard(tp, fp, fn)
	end

	if (conf.gce)
		@inbounds sliceMetricsTupl[6] = MainOverlap.gce(tn, tp, fp, fn)
	end

	if (conf.randInd)
		@inbounds sliceMetricsTupl[7] = RandIndex.calculateAdjustedRandIndex(tn, tp, fp, fn)
	end

	if (conf.kc)
		@inbounds sliceMetricsTupl[8] = ProbabilisticMetrics.calculateCohenCappa(tn, tp, fp, fn)
	end

	if (conf.vol)
		@inbounds sliceMetricsTupl[9] = VolumeMetric.getVolumMetric(tp, fp, fn)
	end

	if (conf.mi)
		@inbounds sliceMetricsTupl[10] = InformationTheorhetic.mutualInformationMetr(tnPrim, tp, fp, fn)
	end

	if (conf.vi)
		@inbounds sliceMetricsTupl[11] = InformationTheorhetic.variationOfInformation(tn, tp, fp, fn)
	end

end
end#TpfpfnKernel