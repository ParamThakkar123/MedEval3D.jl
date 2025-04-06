module WorkQueueUtils
using ..BitWiseUtils, CUDA, Logging, ..CUDAGpuUtils, ..ResultListUtils, ..WorkQueueUtils, Logging, StaticArrays, ..IterationUtils, ..ReductionUtils, ..CUDAAtomicUtils, ..MetaDataUtils, ..BitWiseUtils
export allocateWorkQueue, appendToWorkQueue, @appendToWorkQueueBasic, @appendToLocalWorkQueue, @appendToGlobalWorkQueue
"""
allocate memory for  work queues
	entries means
		1)xMeta
		2)yMeta
		3)zMeta
		4)isGold
 In order to prevent overwriting  we will create 8 separe work queues for each even or odd metax,metay and meta Z ... 
	also initializes counters for each 
	names are encoded like E is about even and O for odd so workQueueEEE will mean that all metaX,metaY and metaZ are even    
   returns  workQueue,workQueueCounter
	
	"""
function allocateWorkQueue(metaDataLength)
	queueSize = cld(metaDataLength * 2, 8) + 2#*2 becouse of gold and segm pass divided by 8 becouse we have 8 work queues
	return (CUDA.zeros(UInt16, 4, (queueSize)), CUDA.zeros(UInt16, (1)))
end

"""
atomically append the block linear index and information is it gold or other pass 
also we need to be sure that we appended to the correct work queue based on the properties of the xMeta,yMeta,zMeta - so are they even, odd ...
"""
macro appendToLocalWorkQueue(metaX, metaY, metaZ, isGold)
	return esc(quote
		old = atomicallyAddOne(workCounterLocalInShmem)
		#CUDA.@cuprint " ooo old $(old) metaX $($metaX) metaY $($metaY) metaZ $($metaZ) isGold $($isGold) workCounterLocalInShmem[1] $(workCounterLocalInShmem[1]) \n"
		@inbounds shmemblockData[old*4+1] = $metaX
		@inbounds shmemblockData[old*4+2] = $metaY
		@inbounds shmemblockData[old*4+3] = $metaZ
		@inbounds shmemblockData[old*4+4] = $isGold
	end)#qote 
end#appendToWorkQueue

"""
the function above is block private now we need to load the work queaue into main work queue
	and additionally atomically add to general work queue counter
"""
macro appendToGlobalWorkQueue()
	return esc(quote
		if (workCounterLocalInShmem[1] > 0)
			#old is a spot where we should start adding new entries
			sync_threads()
			@ifXY 1 1 workCounterHelper[1] = atomicallyAddToSpot(workQueueCounter, 1, workCounterLocalInShmem[1]) - workCounterLocalInShmem[1]
			sync_threads()
			# @ifXY 1 1 CUDA.@cuprint "appendToGlobalWorkQueue old $(old)  workCounterLocalInShmem[1] $(workCounterLocalInShmem[1]) \n "
			#we multiply it by 4 so each thread will have potentially only one data entry to do 
			@iterateLinearlyCheckAll(cld(workCounterLocalInShmem[1] * 4, blockDimX() * blockDimY()), workCounterLocalInShmem[1] * 4, begin
				@inbounds workQueue[(workCounterHelper[1])*4+i] = shmemblockData[i]
			end)
		end
	end)#qote 
end#appendToWorkQueue
end