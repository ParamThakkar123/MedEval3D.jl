"""
new optimazation idea  - try to put all data in boolean arrays in shared memory  when getting means
next we would need only to read shared memory - yet first one need to check wheather there would be enough shmem on device
calculating intercalss correlation
"""
module InterClassCorrKernel

using CUDA, ..CUDAGpuUtils, ..IterationUtils, ..ReductionUtils, ..MemoryUtils, ..CUDAAtomicUtils
export prepareInterClassCorrKernel, calculateInterclassCorr

"""
preparation for the interclass correlation kernel - we prepare the  GPU arrays - in this kernel they are small , calculate required loop iterations 
and using occupancy API calculate optimal number of  threads and blocks for both kernels 
also return arguments lists for both kernels
"""
function prepareInterClassCorrKernel()
	mainArrDims = (2, 2, 2)
	sumOfGold = CUDA.zeros(1)
	sumOfSegm = CUDA.zeros(1)
	sswTotal = CUDA.zeros(1)
	ssbTotal = CUDA.zeros(1)
	numberToLooFor = 1
	totalNumbOfVoxels = (mainArrDims[1] * mainArrDims[2] * mainArrDims[3])
	pixPerSlice = mainArrDims[1] * mainArrDims[2]
	iterLoop = 5
	argsMain = (sumOfGold, sumOfSegm, sswTotal, ssbTotal, iterLoop, pixPerSlice, totalNumbOfVoxels, numberToLooFor)
	get_shmem(threads) = 4 * 33 + 3  #the same for both kernels
	threads, blocks = getThreadsAndBlocksNumbForKernel(get_shmem, kernel_InterClassCorr, (CUDA.zeros(2, 2, 2), CUDA.zeros(2, 2, 2), argsMain...))
	totalNumbOfVoxels = (mainArrDims[1] * mainArrDims[2] * mainArrDims[3])
	pixPerSlice = cld(totalNumbOfVoxels, blocks)
	iterLoop = UInt32(fld(pixPerSlice, threads[1] * threads[2]))

	argsMain = (sumOfGold, sumOfSegm, sswTotal, ssbTotal, iterLoop, pixPerSlice, totalNumbOfVoxels, numberToLooFor)


	return (argsMain, threads, blocks, totalNumbOfVoxels)
end

"""
calculates slicewise and global interclass correlation metric
"""
function calculateInterclassCorr(flatGold, flatSegm, threads, blocks, args, numberToLooFor)::Float64
	mainArrDims = size(flatGold)
	totalNumbOfVoxels = length(flatGold)
	pixPerSlice = cld(totalNumbOfVoxels, blocks)
	iterLoop = UInt32(fld(pixPerSlice, threads[1] * threads[2]))
	#resetting
	for i in 1:4
		CUDA.fill!(args[i], 0)
	end
	argsMain = (args[1], args[2], args[3], args[4], iterLoop, pixPerSlice, totalNumbOfVoxels, numberToLooFor)

	@cuda threads = threads blocks = blocks cooperative = true kernel_InterClassCorr(flatGold, flatSegm, argsMain...)

	# println("grandMean[1] $(grandMean[1])  \n")
	# @cuda threads=threads blocks=blocks  kernel_InterClassCorr(flatGold  ,flatSegm,args... )
	ssw = args[3][1] / args[7]
	ssb = args[4][1] / (args[7] - 1) * 2

	return (ssb - ssw) / (ssb + ssw)

end


"""
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
goldBoolGPU - array holding data of gold standard bollean array
segmBoolGPU - boolean array with the data we want to compare with gold standard
ic - holding single value for global interclass correlation
intermediateRes- array holding slice wise results for ic
loopNumb - number of times the single lane needs to loop in order to get all needed data
sliceEdgeLength - length of edge of the slice we need to square this number to get number of pixels in a slice
amountOfWarps - how many warps we can stick in the block
"""

function kernel_InterClassCorr(flatGold, flatSegm, sumOfGold, sumOfSegm, sswTotal, ssbTotal, iterLoop, pixPerSlice, totalNumbOfVoxels, numberToLooFor)
	grid_handle = this_grid()

	#for storing results from warp reductions
	shmemSum = @cuStaticSharedMem(Float32, (32, 2))   #thread local values that are meant to store some results - like means ... 
	grandMeanShmem = @cuStaticSharedMem(Float32, (1))
	offsetIter = UInt8(1)

	locValA = Float32(0)
	locValB = Float32(0)
	#reset shared memory
	@ifY 1 shmemSum[threadIdxX(), 1] = 0
	@ifY 2 shmemSum[threadIdxX(), 2] = 0
	sync_threads()

	#first we add 1 for each spot we have true - so we will get sum  - and from sum we can get mean
	@iterateLinearlyMultipleBlocks(iterLoop, pixPerSlice, totalNumbOfVoxels, begin
		locValA += flatGold[i] == numberToLooFor
		locValB += flatSegm[i] == numberToLooFor
	end)#for
	@redWitAct(offsetIter, shmemSum, locValA, +, locValB, +)
	sync_threads()
	@addAtomic(shmemSum, sumOfGold, sumOfSegm)
	### sums should be in place
	sync_grid(grid_handle)
	grandMeanShmem[1] = ((sumOfGold[1] / totalNumbOfVoxels) + (sumOfSegm[1] / totalNumbOfVoxels)) / 2
	locValA = 0
	locValB = 0
	@ifY 1 shmemSum[threadIdxX(), 1] = 0
	@ifY 2 shmemSum[threadIdxX(), 2] = 0
	sync_threads()

	@iterateLinearlyMultipleBlocks(iterLoop, pixPerSlice, totalNumbOfVoxels, begin
		m = ((flatGold[i] == numberToLooFor) + (flatSegm[i] == numberToLooFor)) / 2
		locValA += (((flatGold[i] == numberToLooFor) - m)^2) + (((flatSegm[i] == numberToLooFor) - m)^2)
		locValB += ((m - grandMeanShmem[1])^2)
	end)#for
	#now we accumulated ssw and locValB - we need to reduce it
	offsetIter = UInt8(1)
	@redWitAct(offsetIter, shmemSum, locValA, +, locValB, +)
	sync_threads()
	@addAtomic(shmemSum, sswTotal, ssbTotal)
	return nothing

end
end#InterClassCorrKernel
