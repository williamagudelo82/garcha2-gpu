
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
template<class scalar_type>
static __inline__ __device__ vec_type<scalar_type,4> shfl(vec_type<scalar_type, 4> var, int laneMask, int width=warpSize)
{
    vec_type<scalar_type, 4> tmp;
    tmp.x = shfl(var.x, laneMask, width);
    tmp.y = shfl(var.y, laneMask, width);
    tmp.z = shfl(var.z, laneMask, width);
    return tmp;
}



template<class scalar_type, bool compute_energy, bool compute_factor, bool lda>
__global__ void gpu_compute_density(scalar_type* const energy, scalar_type* const factor, const scalar_type* const point_weights,
                                    uint points, const scalar_type* function_values, const vec_type<scalar_type,4>* gradient_values,
                                    const vec_type<scalar_type,4>* hessian_values, uint m, scalar_type* out_partial_density, vec_type<scalar_type,4>* out_dxyz, vec_type<scalar_type,4>* out_dd1, vec_type<scalar_type,4>*  out_dd2)
{

    uint point = blockIdx.x;
    uint i     = threadIdx.x + blockIdx.y * DENSITY_BLOCK_SIZE;
    
    uint min_i = blockIdx.y*DENSITY_BLOCK_SIZE; //Para invertir el loop de bj

    scalar_type partial_density (0.0f);
    vec_type<scalar_type,WIDTH> dxyz, dd1, dd2;

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

    scalar_type Fj_memoria;
    vec_type<scalar_type,4> Fgj_memoria;
    vec_type<scalar_type,4> Fhj1_memoria;
    vec_type<scalar_type,4> Fhj2_memoria;

    scalar_type Fi;
    vec_type<scalar_type,4> Fgi, Fhi1, Fhi2;

    int position = threadIdx.x;

    
    for (int bj = min_i; bj >= 0; bj -= DENSITY_BLOCK_SIZE)
    {
        //Density deberia ser GET_DENSITY_BLOCK_SIZE

        __syncthreads();
        if( bj+position<m )
        {
            Fj_memoria = function_values[(m) * point + (bj+position)];
            if(!lda)
            {
                Fgj_memoria = gradient_values[(m) * point + (bj+position)];
                Fhj1_memoria = hessian_values[(m)*2 * point +(2 * (bj + position) + 0)];
                Fhj2_memoria = hessian_values[(m)*2 * point +(2 * (bj + position) + 1)];
            }
        }


        __syncthreads();
        if(bj==min_i)
        {
            //Fi=fj_sh[position];
            //Fi=  shfl( Fj_memoria, position, WARP_SIZE);
            Fi = Fj_memoria;
            if(!lda)
            {
                Fgi =  Fgj_memoria;
                Fhi1 = Fhj1_memoria;
                Fhi2 = Fhj2_memoria;
            }
        }

        for(int j=0; j<DENSITY_BLOCK_SIZE /*&& bj+j <= i*/; j++)
        {
          
            //fetch es una macro para tex2D
            scalar_type rdm_this_thread = fetch(rmm_input_gpu_tex, (float)(bj+j), (float)i);
            scalar_type Fj_otro = shfl(Fj_memoria, j, WARP_SIZE);
            
            

            w += rdm_this_thread * Fj_otro;
            if(!lda)
            {
                vec_type<scalar_type,4> Fgj_otro = shfl(Fgj_memoria, j ,WARP_SIZE);
                vec_type<scalar_type,4> Fhj1_otro = shfl(Fhj1_memoria, j ,WARP_SIZE);
                vec_type<scalar_type,4> Fhj2_otro = shfl(Fhj2_memoria, j ,WARP_SIZE);
                w3 += Fgj_otro* rdm_this_thread ;
                ww1 += Fhj1_otro * rdm_this_thread;
                ww2 += Fhj2_otro * rdm_this_thread;
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
    if(!valid_thread)
    {
        partial_density=0.0f;
        dd1 = dd2 = dxyz = vec_type<scalar_type, 4>(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    for(int j=2;  j <= DENSITY_BLOCK_SIZE ; j=j*2) // 
    {
        int index=position + DENSITY_BLOCK_SIZE/j;
        partial_density += shfl( partial_density, index, WARP_SIZE);
        dxyz += shfl(dxyz, index,WARP_SIZE);        
        dd1 += shfl(dd1, index,WARP_SIZE);        
        dd2 += shfl(dd2, index,WARP_SIZE);        
    }
    if(threadIdx.x==0)
    {
        const int myPoint = blockIdx.y*points + blockIdx.x;
        out_partial_density[myPoint] = partial_density;
        out_dxyz[myPoint]            = dxyz;
        out_dd1[myPoint]             = dd1;
        out_dd2[myPoint]             = dd2;
    }
}

