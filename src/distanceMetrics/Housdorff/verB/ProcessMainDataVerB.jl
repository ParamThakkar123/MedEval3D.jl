
macro executeDataIterWithPadding()
  #some data cleaning
  locArr::UInt32 = UInt32(0)
  # locFloat::Float32 = Float32(0.0)
  isMaskFull::Bool= true
  isMaskOkForProcessing::Bool = true
  offset = 1
  ############## upload data
  @loadMainValues                                        
  sync_threads()
  ########## check data aprat from padding
  #can be skipped if we have the block with already all results analyzed 
  if(getIsTotalFPorFNnotYetCovered(resshmem )   )
      @validateData() 
  end                  
  #processing padding
  @processPadding()
  #now in case to establish is the block full we need to reduce the isMaskFull information
  @establishIsFull()
  
end#executeDataIterWithPadding

"""
we are processing padding 
"""
macro processPadding()
    #so here we utilize iter3 with 1 dim fixed 
    @unroll for  dim in 1:3, numb in [1,34]              
        @iter3dFixed dim numb if( isPaddingToBeAnalyzed(resShmem,dim,numb ))
                if(val)#if value in padding associated with this spot is true
                    # setting value in global memory
                    @inbounds  analyzedArr[x,y,z]= true
                    # if we are here we have some voxel that was false in a primary mask and is becoming now true - if it is additionaly true in reference we need to add it to result
                    resShmem[1,1,1]=true
                    if(@inbounds referenceArray[x,y,z])
                        appendResultPadding(metaData, linIndex, x,y,z,iterationnumber, dim,numb)
                end#if
                #we need to check is next block in given direction exists
                if(isNextBlockExists(metaData,dim, numb ,linIter, isPassGold, maxX,maxY,maxZ))
                    if(resShmem[1,1,1])
                        @ifXY 1 1 setAsToBeActivated(metaData,linIndex,isPassGold)
                    end    
                end # if isNextBlockExists       
            end # if isPaddingToBeAnalyzed   
        end#iter3dFixed       
    end#for

end



"""
we need to establish is the block full after dilatation step 
"""
macro establishIsFull()
    @redWitAct(offsetIter,shmemSum,  isMaskFull,& )
    sync_threads()
    #now if it evaluated to 1 we should save it to metadata 
    @ifXY 2 2 setBlockAsFull(metaData,linIndex, isGoldPass)
end#establishIsFull



"""
   loads main values from analyzed array into shared memory and to locArr - which live in registers   
   it all works under the assumption that x and y dimension of the thread block and data block is the same           
"""                
                
macro loadMainValues()
        @iter3dWithVal  dataBlockDims loopX loopY loopZ blockBeginingX blockBeginingY blockBeginingZ analyzedArr begin
        #val is given by macro as value of this x,y,z 
        locArr|= val << (zIter-1)
        processMaskData( val, zIter, resShmem) 
        #zIter given in macro as we are iterating in this spot
        #we add to source shmem also becouse 
        sourceShmem[threadIdxX(), threadIdxY(), zIter]                
    end                
end #loadMainValues
                
                
"""
 validates data is of our intrest               
"""                
macro validateData()
    @iter3dW  dataBlockDims loopX loopY loopZ blockBeginingX blockBeginingY blockBeginingZ resShemVal begin
        locVal::Bool = @inbounds  (locArr>>(zIter-1) & 1)
        resShemVal::Bool = @inbounds resShmem[threadIdxX()+1,threadIdxY()+1,zIter+1]             
        locValOrShmem = (locVal | resShemVal)
        #those needed to establish weather data block will remain active
        isMaskFull= locValOrShmem & isMaskFull
        @ifverr zzz isMaskEmpty = ~locValOrShmem & isMaskEmpty
        if(!locVal && resShemVal)       
              innerValidate(analyzedArr,referenceArray,x,y,z,privateResArray,privateResCounter,iterationnumber,sourceShmem  )
        end#if
     end#3d iter 
    
    
 end  #validateData                  

"""
this will be invoked when we know that we have a true in a spot that was false before this dilatation step and its task is to set to true appropriate spot in global array
- so proper dilatation
check weather we have true also in reference array - if so we  need to add this spot to the block result list in case we are invoke it from padding we need to look even futher into the
next block data to establish could this spot be activated from there
"""
  function innerValidate(analyzedArr,referenceArray,x,y,z,privateResArray,privateResCounter,iterationnumber,sourceShmem  )
            # setting value in global memory
            @inbounds  analyzedArr[x,y,z]= true
            # if we are here we have some voxel that was false in a primary mask and is becoming now true - if it is additionaly true in reference we need to add it to result
    
            if(@inbounds referenceArray[x,y,z])
                #results now are stored in a matrix where first 3 entries are x,y,z coordinates entry 4 is in which iteration we covered it and entry 5 from which direction - this will be used if needed        
                #privateResCounter privateResArray are holding in metadata blocks results and counter how many results were already added 
                #in each thread block we will have separate rescounter, and res array for goldboolpass and other pass
               direction=  getDir(sourceShmem)   
               appendResultMainPart(metaData, linIndex, x,y,z,iterationnumber, direction)
            end#if
  end#innerValidate 
     



"""
Help to establish should we validate the voxel - so if ok add to result set, update the main array etc
  in case we have some true in padding
  generally we need just to get idea if
    we already had true in this very spot - if so we ignore it
    can this spot be reached by other voxels from the block we are reaching into - in other words padding is analyzing the same data as other block is analyzing in its main part
      hence if the block that is doing it in main part will reach this spot on its own we will ignore value from padding 

  in order to reduce sears direction by 1 it would be also beneficial to know from where we had came - from what direction the block we are spilled into padding 
"""
function isPaddingValToBeValidated(dir,analyzedArr, x,y,z )::Bool
     
if(dir!=5)  if( @inbounds resShmem[threadIdxX(),threadIdxY(),zIter-1]) return false  end end #up
if(dir!=6)  if( @inbounds  resShmem[threadIdxX(),threadIdxY(),zIter+1]) return false  end  end #down
    
if(dir!=1)   if( @inbounds  resShmem[threadIdxX()-1,threadIdxY(),zIter]) return false  end  end #left
if(dir!=2)   if( @inbounds   resShmem[threadIdxX()+1,threadIdxY(),zIter]) return false  end end  #right

if(dir!=4)   if(  @inbounds  resShmem[threadIdxX(),threadIdxY()+1,zIter]) return false  end  end #front
if(dir!=3)   if( @inbounds  resShmem[threadIdxX(),threadIdxY()-1,zIter]) return false  end end  #back
  #will return true only in case there is nothing around 
  return true
end


"""
uploaded data from shared memory in amask of intrest gets processed in this function so we need to  
    - save it to registers (to locArr)
    - save to the 6 surrounding voxels in shared memory intermediate results 
            - as we also have padding we generally start from spot 2,2 as up and to the left we have 1 padding
            - also we need to make sure that in corner cases we are getting to correct spot
"""
function processMaskData(maskBool::Bool
                         ,zIter::UInt8
                         ,resShmem
                          ) #::CUDA.CuRefValue{Int32}
    # save it to registers - we will need it later
    #locArr[zIter]=maskBool
    #now we are saving results evrywhere we are intrested in so around without diagonals (we use supremum norm instead of euclidean)
    #locArr.x|= maskBool << zIter
    if(maskBool)
        @inbounds resShmem[threadIdxX()+1,threadIdxY()+1,zIter]=true #up
        @inbounds  resShmem[threadIdxX()+1,threadIdxY()+1,zIter+2]=true #down
    
        @inbounds  resShmem[threadIdxX(),threadIdxY()+1,zIter+1]=true #left
        @inbounds   resShmem[threadIdxX()+2,threadIdxY()+1,zIter+1]=true #right

        @inbounds  resShmem[threadIdxX()+1,threadIdxY()+2,zIter+1]=true #front
        @inbounds  resShmem[threadIdxX()+1,threadIdxY(),zIter+1]=true #back
    end#if    
    
end#processMaskData

"""
now in case we  want later to establish source of the data - would like to find the true distances  not taking the assumption of isometric voxels
we need to store now data from what direction given voxel was activated what will later gratly simplify the task of finding the true distance 
we will record first found true voxel from each of six directions 
                 top 6 
                bottom 5  
                left 2
                right 1 
                anterior 3
                posterior 4
"""
function getDir(sourceShmem)
if( @inbounds sourceShmem[threadIdxX(),threadIdxY(),zIter-1]) return 6  end  #up
if( @inbounds  sourceShmem[threadIdxX(),threadIdxY(),zIter+1]) return 5  end #down

if( @inbounds  sourceShmem[threadIdxX()-1,threadIdxY(),zIter]) return 2  end #left
if( @inbounds   sourceShmem[threadIdxX()+1,threadIdxY(),zIter]) return 1  end #right

if(  @inbounds  sourceShmem[threadIdxX(),threadIdxY()+1,zIter]) return 3  end #front
 if( @inbounds  sourceShmem[threadIdxX(),threadIdxY()-1,zIter]) return 4  end #back
end#getDir