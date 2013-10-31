
#if FULL_DOUBLE
static __inline__ __device__ double fetch_double(texture<int2, 2> t, float x, float y)
{
    int2 v = tex2D(t,x,y);
    return __hiloint2double(v.y, v.x);
}   
#define fetch(t,x,y) fetch_double(t,x,y)
#else
#define fetch(t,x,y) tex2D(t,x,y)
#endif
static __inline__ __device__ float shfl(float var, int laneMask, int width=warpSize)
{
    return __shfl(var,laneMask,width);
}
static __inline__ __device__ double shfl(double var, int laneMask, int width=warpSize)
{
    int hi, lo;
    asm volatile( "mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "d"(var) );
    hi = __shfl( hi, laneMask, width );
    lo = __shfl( lo, laneMask, width );
    return __hiloint2double( hi, lo );
}

template<class scalar_type, bool compute_energy, bool compute_factor, bool lda>
__global__ void gpu_compute_density(scalar_type* const energy, scalar_type* const factor, const scalar_type* const point_weights,
                                    uint points, const scalar_type* function_values, const vec_type<scalar_type,4>* gradient_values,
                                    const vec_type<scalar_type,4>* hessian_values, uint m, uint block_height, scalar_type* out_partial_density, vec_type<scalar_type,4>* out_dxyz, vec_type<scalar_type,4>* out_dd1, vec_type<scalar_type,4>*  out_dd2)
{

    uint point = blockIdx.x;
    uint i     = threadIdx.x + blockIdx.y * (DENSITY_BLOCK_SIZE*block_height) + threadIdx.y*DENSITY_BLOCK_SIZE;
     
    uint min_i = blockIdx.y * (DENSITY_BLOCK_SIZE*block_height) + threadIdx.y*DENSITY_BLOCK_SIZE; //Para invertir el loop de bj
    if((threadIdx.y + blockDim.y * blockIdx.y) < block_height)
    {

    scalar_type partial_density (0.0f);
    vec_type<scalar_type,WIDTH> dxyz, dd1, dd2;
    dxyz=dd1=dd2 =vec_type<scalar_type,4>(0.0f,0.0f,0.0f,0.0f);

    if (!lda)
    {
        dxyz = dd1 = dd2 = vec_type<scalar_type,4>(0.0f,0.0f,0.0f,0.0f);
    }

    bool valid_thread = ( i < m );


    scalar_type w = 0.0f;
    vec_type<scalar_type,4> w3, ww1, ww2;
    if (!lda)
    {
        w3 = ww1 = ww2 = vec_type<scalar_type,4>(0.0f,0.0f,0.0f,0.0f);
    }

    scalar_type Fi;
    vec_type<scalar_type,4> Fgi, Fhi1, Fhi2;
    scalar_type Fj_memoria;

    int position = threadIdx.x;
    int position_offset= position + DENSITY_BLOCK_SIZE*threadIdx.y;

//    __shared__ scalar_type fj_sh[DENSITY_BLOCK_SIZE];
    __shared__ vec_type<scalar_type, WIDTH> fgj_sh [DENSITY_BLOCK_SIZE*2];
    __shared__ vec_type<scalar_type, WIDTH> fh1j_sh [DENSITY_BLOCK_SIZE*2];
    __shared__ vec_type<scalar_type, WIDTH> fh2j_sh [DENSITY_BLOCK_SIZE*2];

    for (int bj = min_i; bj >= 0; bj -= DENSITY_BLOCK_SIZE)
    {
        //Density deberia ser GET_DENSITY_BLOCK_SIZE

        __syncthreads();
        if( bj+position<m )
        {

            Fj_memoria = function_values[(m) * point + (bj+position)];
            if(!lda)
            {
                fgj_sh[position_offset] = gradient_values[(m) * point + (bj+position)];

                fh1j_sh[position_offset] = hessian_values[(m)*2 * point +(2 * (bj + position) + 0)];
                fh2j_sh[position_offset] = hessian_values[(m)*2 * point +(2 * (bj + position) + 1)];
            }
        }


        __syncthreads();
        if(bj==min_i)
        {
            Fi=Fj_memoria;
            if(!lda)
            {
                Fgi = fgj_sh[position_offset];
                Fhi1 = fh1j_sh[position_offset] ;
                Fhi2 = fh2j_sh[position_offset] ;
            }
        }

    //    if(valid_thread)
        {
            for(int j=0; j<DENSITY_BLOCK_SIZE /*&& bj+j <= i*/; j++)
            {
              
                //fetch es una macro para tex2D
                scalar_type rdm_this_thread = fetch(rmm_input_gpu_tex, (float)(bj+j), (float)i);

                w += rdm_this_thread * shfl(Fj_memoria, j, WARP_SIZE);

                if(!lda)
                {
                    w3 += fgj_sh[j+threadIdx.y*DENSITY_BLOCK_SIZE]* rdm_this_thread ;
                    ww1 += fh1j_sh[j+threadIdx.y*DENSITY_BLOCK_SIZE] * rdm_this_thread;
                    ww2 += fh2j_sh[j+threadIdx.y*DENSITY_BLOCK_SIZE] * rdm_this_thread;
                }

            }
        }
    }
    if(valid_thread)
    {
        partial_density = Fi * w;
        //TODO: Insertar aca funcion que convierte <,4> a <,3>
        if (!lda)
        {
            dxyz += Fgi * w + w3 * Fi;
            dd1 += Fgi * w3 * 2.0f + Fhi1 * w + ww1 * Fi;

            vec_type<scalar_type,4> FgXXY(Fgi.x, Fgi.x, Fgi.y, 0.0f);
            vec_type<scalar_type,4> w3YZZ(w3.y, w3.z, w3.z, 0.0f);
            vec_type<scalar_type,4> FgiYZZ(Fgi.y, Fgi.z, Fgi.z, 0.0f);
            vec_type<scalar_type,4> w3XXY(w3.x, w3.x, w3.y, 0.0f);

            dd2 += FgXXY * w3YZZ + FgiYZZ * w3XXY + Fhi2 * w + ww2 * Fi;
        }
    }


    __syncthreads();
    //Estamos reutilizando la memoria shared por block para hacer el acumulado por block.
    if(valid_thread)
    {
//        fj_sh[position]=partial_density;
        fgj_sh[position_offset]=dxyz;
        fh1j_sh[position_offset]=dd1;
        fh2j_sh[position_offset]=dd2;
    }
    else
    {
       // partial_density =scalar_type(0.0f);
        fgj_sh[position_offset]=vec_type<scalar_type,4>(0.0f,0.0f,0.0f,0.0f);
        fh1j_sh[position_offset]=vec_type<scalar_type,4>(0.0f,0.0f,0.0f,0.0f);
        fh2j_sh[position_offset]=vec_type<scalar_type,4>(0.0f,0.0f,0.0f,0.0f);
    }
    __syncthreads();

    for(int j=2;  j <= DENSITY_BLOCK_SIZE ; j=j*2) // 
    {
        int index=position_offset + DENSITY_BLOCK_SIZE/j;
         partial_density += shfl( partial_density, index, WARP_SIZE);
        if( position < DENSITY_BLOCK_SIZE/j)
        {
            //fj_sh[position]      += fj_sh[index];
            fgj_sh[position_offset]     += fgj_sh[index];
            fh1j_sh[position_offset]    += fh1j_sh[index];
            fh2j_sh[position_offset]    += fh2j_sh[index];
        }
    }
    if(threadIdx.x==0)
    {
      

        const int myPoint =( blockIdx.y*blockDim.y+threadIdx.y)*points + blockIdx.x;
        out_partial_density[myPoint] = partial_density;
        out_dxyz[myPoint]            = fgj_sh[position_offset];
        out_dd1[myPoint]             = fh1j_sh[position_offset];
        out_dd2[myPoint]             = fh2j_sh[position_offset];
    }
}
}
