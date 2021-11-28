


"""
utility functions helping managing result list 
"""
module ResultListUtils
using CUDA,Main.MetaDataUtils, Main.CUDAAtomicUtils
export getResLinIndex,allocateResultLists,@addResult

"""
allocate memory on GPU for storing result list 
    totalFpCount-  total number of false positives
    TotalFNCount - total number of false negatives

    in the array first 3 entries will be x,y,z than isGold - 1 if it is  related to gold pass dilatations
        , direction from which result was set and the iteration number in which it was covered
    in the second list we have  the UInt32 Ids generated by the  getResLinIndex function   
"""
function allocateResultLists(totalFpCount,TotalFNCount)
  return CUDA.zeros(UInt16, (totalFpCount+ TotalFNCount+1),6 )
end#allocateResultList




 """
 adding result to the result list at correct postion using data from metaData - so we get from the metadata offset and result counter 
 metadata - 4 dimensional array holding metaData
 xMeta,yMeta,zMeta -  x,y,z coordinates of block of intrest in meta Data
 resList - list of result (matrix to be more precise) where we will wrtie the results
 resListIndicies - list of indicies related to results
 x,y,z - coordinates where we found point of intrest 
 dir - direction from which dilatation covering this voxel had happened
 queueNumber - what fp or fn queue we are intrested in modyfing now 
 metaDataDims - dimensions of metadata array
mainArrDims - dimensions of main array
 isGold - indicated is this a gold dilatation step (then it will evaluate to 1 otherwise 0 )
 """
 macro addResult(metaData ,xMeta,yMeta,zMeta, resList,x,y,z, dir,iterNumb,queueNumber,metaDataDims,mainArrDims ,isGold  )
  return esc(quote

# linearIndex = ($xMeta+1) + ($yMeta)*$metaDataDims[1] + ($zMeta)*$metaDataDims[1]*$metaDataDims[2] + (getNewCountersBeg()+$queueNumber-1)*$metaDataDims[1]*$metaDataDims[2]*$metaDataDims[3]
  resListPos = ($metaData[($xMeta+1),($yMeta+1),($zMeta+1), (getResOffsetsBeg() +$queueNumber) ]+atomicallyAddToSpot( metaData,linearIndex,UInt32(1) ))+1
# qn = $queueNumber
# xm = $xMeta
# ym = $yMeta
# zm = $zMeta
# xx = $x
# yy= $y
# zz=$z
# dd= $dir
# if(qn==1)
#  CUDA.@cuprint "\n resListPos $(resListPos) queueNumber $(qn)  xMeta $(xm) yMeta $(ym)  zMeta $(zm) x $(xx) y $(yy) z $(zz) linearIndex $(linearIndex) dd $(dd)   \n "
# end
@inbounds $resList[ resListPos, 1]=$x 
@inbounds $resList[ resListPos, 2]=$y 
@inbounds $resList[ resListPos, 3]=$z 
@inbounds $resList[ resListPos, 4]= $isGold
@inbounds $resList[ resListPos, 5]= $dir
@inbounds $resList[ resListPos, 6]= $iterNumb
# @inbounds $resListIndicies[resListPos]=getResLinIndex($x,$y,$z,$isGold,$mainArrDims)
# CUDA.@cuprint "\n linIndex $(getResLinIndex(x,y,z,isGold,mainArrDims))  \n "
#addResHelper(resListIndicies,resListPos,x)
 end)#quote
 end#addResult



 """
 giver the result row that holds data about covered point and in what iteration, from what direction and in what pass it was covered
 resRow - array where first 3 entries are x,y,z positions then is gold,
 """
  function getResLinIndex(x,y,z,isGold,mainArrDims)::UInt32
     # last one is in order to differentiate between gold pass and other pass ...
     return x+ y*mainArrDims[1]+ z* mainArrDims[1]*mainArrDims[2]+ isGold*mainArrDims[1]*mainArrDims[2]*mainArrDims[3]  
   end#getResLinIndex
 

end#ResultListUtils