module ScanForDuplicates
using CUDA, Logging, ..CUDAGpuUtils, ..WorkQueueUtils, ..ScanForDuplicates, Logging, StaticArrays, ..IterationUtils, ..ReductionUtils, ..CUDAAtomicUtils, ..MetaDataUtils
export @getIsToVal, @loadAndScanForDuplicates, @setIsToBeValidated, @scanForDuplicatesB, scanForDuplicatesMainPart, scanWhenDataInShmem, manageDuplicatedValue

"""
as we are operating under assumption that we do not know how many warps we have - we do not know the y dimension of thread block we need to load data into registers with a loop 
and within the same loop scan it for duplicates
so if we have more than 12 warps we will execute the loop once - in case we have more we need to execute it in the loop
iterThrougWarNumb - indicates how many times we need to  iterate to cover all 12 ques if we have at least 12 warps available (ussually we will) we will execute it once
locArr,offsetIter,localOffset - variables used for storing some important constants

"""
macro loadAndScanForDuplicates(iterThrougWarNumb, locArr, offsetIter)

	return esc(quote
		@unroll for outerWarpLoop in 0:$iterThrougWarNumb
			#represents the number of queue if we have enought warps at disposal it equals warp number so idY
			innerWarpNumb = (threadIdxY() + outerWarpLoop * blockDimY())
			#at this point we have actual counters with correction for duplicated values  we can compare it with the  total values of fp or fn of a given queue  if we already covered
			#all points of intrest there is no point to futher analyze this block or padding

			@setIsToBeValidated()

			if (innerWarpNumb < 15)
				shmemSum[34, innerWarpNumb] = 0
				shmemSum[35, innerWarpNumb] = 0
				shmemSum[36, innerWarpNumb] = 0
			end
		end#for    
		@exOnWarp 15 if (isInRange)

			@setMeta((getIsToBeAnalyzedNumb() + 15), (@getIsToVal(1) || @getIsToVal(3) || @getIsToVal(5) || @getIsToVal(7) || @getIsToVal(11) || @getIsToVal(13)))#sourceShmem[(threadIdxX())+33*8]
		end
		@exOnWarp 16 if (isInRange)

			@setMeta((getIsToBeAnalyzedNumb() + 16), (@getIsToVal(2) || @getIsToVal(4) || @getIsToVal(6) || @getIsToVal(8) || @getIsToVal(10) || @getIsToVal(12)))#sourceShmem[(threadIdxX())+33*6] 
		end
		sync_threads()


	end)#quote



end #loadAndScanForDuplicates    

"""
loads value from res shmem about weather a queue with supplied numb has anything worth validating
"""
macro getIsToVal(numb)
	return esc(quote
		(shmemPaddings[(threadIdxX())+($numb+21)*33])
	end)
end

"""
   here we will mark in metadata weather there is anything to be verified - here in given que ie - weather it is possible in given queue to cover anything more in next dilatation step
   so it is important for analysisof this particular block is  there is true - is there is non 0 amount of points to cover in any queue of the block
   simultaneously the border queues should indicate for neighbouring blocks (given they exist ) is there is any point in analyzing the paddings ...
	  so we need this data in 3 places 
	  1) for the getIsToBeAnalyzedNumb value in metadata of blocks around
	  2) for isNotTobeAnalyzed for a current block
	  3) also we should accumulate values pf counters - add and reduce across all blocks of tps and fps covered - so we will know when to finish the dilatation steps
"""
macro setIsToBeValidated()
	return esc(quote
		@exOnWarp innerWarpNumb begin
			if (innerWarpNumb < 15 && isInRange)
				#we need also to remember that data wheather there are any futher points of intrest is not only in the current block
				# so here we establish what are the coordinates of metadata of intrest so for example  our left fp and left FN are of intrest to block to the left ...
				newXmeta = xMeta + (-1 * (innerWarpNumb == 1 || innerWarpNumb == 2)) + (innerWarpNumb == 3 || innerWarpNumb == 4) + 1
				newYmeta = yMeta + (-1 * (innerWarpNumb == 5 || innerWarpNumb == 6)) + (innerWarpNumb == 7 || innerWarpNumb == 8) + 1
				newZmeta = zMeta + (-1 * (innerWarpNumb == 9 || innerWarpNumb == 10)) + (innerWarpNumb == 11 || innerWarpNumb == 12) + 1
				# #check are we in range 
				if (newXmeta > 0 && newYmeta > 0 && newZmeta > 0 && newXmeta <= metaDataDims[1] && newYmeta <= metaDataDims[2] && newZmeta <= metaDataDims[3] && innerWarpNumb < 13)
					metaData[newXmeta, newYmeta, newZmeta, getIsToBeAnalyzedNumb()+innerWarpNumb] = (shmemPaddings[(threadIdxX())+(innerWarpNumb+21)*33])

				end #if in meta data range
			end#if   
		end#ex on warp    

		#here we set the information weather any of the queue related to fp or fn in a particular block  has still something to be analyzed 
	end)#quote
end#setIsToBeValidated
end#module