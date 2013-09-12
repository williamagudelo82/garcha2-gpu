template<class scalar_type, bool compute_energy, bool compute_factor, bool lda>
__device__ void gpu_compute_density(scalar_type* const energy, scalar_type* const factor, const scalar_type* const point_weights,
                                    uint points, const scalar_type* rdm, const scalar_type* function_values, const vec_type<scalar_type,4>* gradient_values,
                                    const vec_type<scalar_type,4>* hessian_values, uint m, scalar_type& partial_density, vec_type<scalar_type,4>& dxyz, vec_type<scalar_type,4>& dd1, vec_type<scalar_type,4>& dd2)
{
/** New Code **/

  uint point = blockIdx.x;
  uint i     = threadIdx.x;
  

  partial_density = 0.0f;
  if (!lda) { dxyz = dd1 = dd2 = vec_type<scalar_type,4>(0.0f,0.0f,0.0f,0.0f); }
/** New Code **/  

    bool valid_thread = (point < points) && ( i < m );
 

  //scalar_type point_weight;
  //if (valid_thread) point_weight = point_weights[point];

  //__shared__ scalar_type rdm_sh[DENSITY_BATCH_SIZE];

  /***** compute density ******/
  //for (uint i = 0; i < m; i++) { //Este for desaparece
    {
    scalar_type w = 0.0f;
    vec_type<scalar_type,4> w3, ww1, ww2;
    if (!lda) { w3 = ww1 = ww2 = vec_type<scalar_type,4>(0.0f,0.0f,0.0f,0.0f); }

    scalar_type Fi;
    vec_type<scalar_type,4> Fgi, Fhi1, Fhi2;

    //TODO: Cada thread del güarp trae su Fi.
    if (valid_thread) {
      Fi = function_values[COALESCED_DIMENSION(points) * i + point]; //Con la paralelizacion a nivel de thread, esta coalescencia desaparece. Hay que darlo vuelta / transpose.
      if (!lda) {
        Fgi = gradient_values[COALESCED_DIMENSION(points) * i + point];  //Deberia ser: Coalesced_dimension(i) * point + i 
        Fhi1 = hessian_values[COALESCED_DIMENSION(points) * (2 * i + 0) + point];   //Hay que cambiarlo de functions.h
        Fhi2 = hessian_values[COALESCED_DIMENSION(points) * (2 * i + 1) + point];
      }
    }
    #define WARP_SIZE 32 //TODO: Cambiar por la constante de warpsize correcta (o sea, la constante general)
    int position_in_warp = threadIdx.x % WARP_SIZE;
    __shared__ scalar_type fj_sh[WARP_SIZE];
    __shared__ vec_type<scalar_type, WIDTH> fgj_sh [WARP_SIZE];
    __shared__ vec_type<scalar_type, WIDTH> fh1j_sh [WARP_SIZE];
    __shared__ vec_type<scalar_type, WIDTH> fh2j_sh [WARP_SIZE];
    for (int bj = 0; bj <= i; bj += WARP_SIZE) { //Density deberia ser GET_WARP_SIZE
        /*
        fj_sh[warp_size]

        fj_sh[]=fj[j,punto]
        for jj=  0 .. warp_size 
        {
            j = jj + bj
            leer rdm(i,j)
            w+=fj*ci(j)
        }
         */
        scalar_type valid_position_in_warp = (1.0f * (bj+position_in_warp<=i)) ;
        fj_sh[position_in_warp] = function_values[COALESCED_DIMENSION(points) * (bj + position_in_warp) + point] * valid_position_in_warp;
        if(!lda)
        {
            fgj_sh[position_in_warp] = vec_type<scalar_type, WIDTH> (gradient_values[COALESCED_DIMENSION(points) * (bj + position_in_warp) + point] *valid_position_in_warp);

            fh1j_sh[position_in_warp] = vec_type<scalar_type,WIDTH>(hessian_values[COALESCED_DIMENSION(points) * (2 * (bj + position_in_warp) + 0) + point]*valid_position_in_warp);
            fh2j_sh[position_in_warp] = vec_type<scalar_type,WIDTH>(hessian_values[COALESCED_DIMENSION(points) * (2 * (bj + position_in_warp) + 1) + point]*valid_position_in_warp);
        }
        __syncthreads();

        for(int j=0; j<WARP_SIZE && bj+j <= i; j++){            
            scalar_type rdm_this_thread = rdm[COALESCED_DIMENSION(m) * i + (bj+j)];
            w += rdm_this_thread * fj_sh[j];

            if(!lda)
            {
                w3 += fgj_sh[j]* rdm_this_thread ;
                ww1 += fh1j_sh[j] * rdm_this_thread;
                ww2 += fh2j_sh[j] * rdm_this_thread;
            }
        }

/*
        __syncthreads();
      if (threadIdx.x < DENSITY_BATCH_SIZE) {
        if (bj + threadIdx.x <= i) rdm_sh[threadIdx.x] = rdm[COALESCED_DIMENSION(m) * i + (bj + threadIdx.x)]; // TODO: uncoalesced. invertir triangulo?
        else rdm_sh[threadIdx.x] = 0.0f;
      }
      __syncthreads();

      if (valid_thread) {
        for (uint j = 0; j < DENSITY_BATCH_SIZE && (bj + j) <= i; j++) {
          float Fj = function_values[COALESCED_DIMENSION(points) * (bj + j) + point];
          w += rdm_sh[j] * Fj;

          if (!lda) {
            vec_type<scalar_type,4> Fgj = gradient_values[COALESCED_DIMENSION(points) * (bj + j) + point];
            w3 += Fgj * rdm_sh[j];

            vec_type<scalar_type,4> Fhj1 = hessian_values[COALESCED_DIMENSION(points) * (2 * (bj + j) + 0) + point];
            vec_type<scalar_type,4> Fhj2 = hessian_values[COALESCED_DIMENSION(points) * (2 * (bj + j) + 1) + point];
            ww1 += Fhj1 * rdm_sh[j];
            ww2 += Fhj2 * rdm_sh[j];
          }
        }
      }
      */
    }

    partial_density = Fi * w;
    //TODO: Insertar aca funcion que convierte <,4> a <,3>
    if (!lda) {
      dxyz += Fgi * w + w3 * Fi;
      dd1 += Fgi * w3 * 2.0f + Fhi1 * w + ww1 * Fi;

      vec_type<scalar_type,4> FgXXY(Fgi.x, Fgi.x, Fgi.y, 0.0f);
      vec_type<scalar_type,4> w3YZZ(w3.y, w3.z, w3.z, 0.0f);
      vec_type<scalar_type,4> FgiYZZ(Fgi.y, Fgi.z, Fgi.z, 0.0f);
      vec_type<scalar_type,4> w3XXY(w3.x, w3.x, w3.y, 0.0f);

      dd2 += FgXXY * w3YZZ + FgiYZZ * w3XXY + Fhi2 * w + ww2 * Fi;
    }
  }
}

