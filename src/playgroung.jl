# 
    using CUDA,Cthulhu

    function reduce_grid_atomic_shmem(op, a::AbstractArray{T},a2::AbstractArray{T} , b) where {T}
        elements = blockDim().x*2
        thread = threadIdx().x
        block = blockIdx().x
        offset = (block-1) * elements
    
        # shared mem to buffer memory loads
        shared = @cuStaticSharedMem(T, (2048,))
        @inbounds shared[thread] = a[offset+thread]
        @inbounds shared[thread+blockDim().x] = a[offset+thread+blockDim().x]
    
        # parallel reduction of values in a block
        d = 1
        while d < elements
            sync_threads()
            index = 2 * d * (thread-1) + 1
            @inbounds if index <= elements && offset+index+d <= length(a)
                shared[index] = op(shared[index], shared[index+d])
            end
            d *= 2
        end
    
        # atomic reduction
        if thread == 1
            @atomic b[] = op(b[], shared[1])
        end
    
        return
    end
    
    function my_sum(a::AbstractArray{T},a2::AbstractArray{T}) where {T}
        b = CUDA.zeros(T, 1)
    
        kernel = @cuda launch=false reduce_grid_atomic_shmem(+, a,a2, b)
    
        config = launch_configuration(kernel.fun)
        threads = min(config.threads, length(a))
        blocks = cld(length(a), threads*2)
    
        @cuda threads=threads blocks=blocks reduce_grid_atomic_shmem(+, a,a2, b)
    
        CUDA.@allowscalar b[]
    end
    

        a = CUDA.ones(100, 100)
        a2 = CUDA.ones(100, 100)
        my_sum(a,a2)



        # sum(a)
        # @assert my_sum(a) ≈ sum(a)















        using CUDA,Cthulhu
# arr1=  CUDA.ones(30,30,30) ;  # 3 dim array of ones
# arr2=  CUDA.ones(30,30,30).*2; # 3 dim array of two’s
# res = CUDA.zeros(30,30,30); # preallocation of memory



# function getFromFourGimTwoDim(shmem::CuDeviceArray{Float32, 4, 1}, source::CuDeviceArray{Float32, 4, 1}, i::Int64,j::Int64, z::Int64, r::Int64 )
#     @inbounds shmem[i,j,z,r] = source[i,j,z]
# end#getFromFourGimTwoDim    

# function myGetIndex(shmem::CuDeviceArray{Float32, 4, 1},  i::Int64,j::Int64, z::Int64, r::Int64 )::Float32
#     return shmem[i,j,z,r]
# end#myGetIndex

# """
# based on https://github.com/JuliaGPU/CUDA.jl/blob/master/examples/pairwise.jl
# """
# function kernelFunction(arr1::CuDeviceArray{Float32, 3, 1}, arr2::CuDeviceArray{Float32, 3, 1}, res::CuDeviceArray{Float32, 3, 1}, n::Int64)
# # getting all required indexes
# i::Int64 = (blockIdx().x-1) * blockDim().x + threadIdx().x
# j::Int64 = (blockIdx().y-1) * blockDim().y + threadIdx().y
# z::Int64 = (blockIdx().z-1) * blockDim().z + threadIdx().z  
# if i <= n && j <= n && z <= n
#     # store to shared memory here I got the size  of the block  so all needed data will be put here 
#     shmem = @cuStaticSharedMem(Float32, (9, 9, 9,2,))
#     #what is important here we are populating shared memory  also in parallel hence this will be execute by each thread 
#     @inbounds shmem[i,j,z,1] = arr1[i,j,z]
#     @inbounds shmem[i,j,z,2] = arr2[i,j,z]


#     # shmem[i,j,z,1] = arr1[i,j,z]
#     # shmem[i,j,z,2] = arr2[i,j,z]
#     #now we sync so the shared memory is available 
#     sync_threads() 
#     # load from shared memory
#     @inbounds res[i, j,z] = shmem[i,j,z,1]+shmem[i,j,z,2]
# end

# return
# end




#     # # calculate a 2D block size from the suggested 1D configuration
#     # # NOTE: we want our launch configuration to be as square as possible,
#     # #       because that minimizes shared memory usage
#     # function get_threads(threads)
#     #     threads_x = floor(Int, sqrt(threads))
#     #     threads_y = threads ÷ threads_x
#     #     return (threads_x, threads_y)
#     # end


#     n= length(arr2)
#     function get_threads(threads)
#         roott = cbrt(threads)
#         return (roott, roott,roott)
#     end



#     # calculate the amount of dynamic shared memory for a 2D block size
#     get_shmem(threads) = 2 * sum(threads) * sizeof(Float32)


#  #@device_code_warntype interactive=true @cuda kernelFunction(arr1, arr2, res, n)


#     kernel = @cuda launch=false kernelFunction(arr1, arr2, res, n)
#     config = launch_configuration(kernel.fun, shmem=threads->get_shmem((9,9,9)))
   
   
#     @cuda threads=9*9*9 blocks=2 kernelFunction(arr1, arr2, res, n)
#     zz= zeros(30,30,30)
#     copyto!(res,zz)

# maximum(res)

#     function get_threadsOld(threads)
#         threads_x = floor(Int, sqrt(threads))
#         threads_y = threads ÷ threads_x
#         return (threads_x, threads_y)
#     end

#     get_threadsOld(17)


