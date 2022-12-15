void pa1_hist_par(int nelts, int nbins, int *__restrict__ data, int *__restrict__ hist) 
{
#pragma omp parallel
 {
//   #pragma omp master
//   {
   int i;
   int hist_private[nbins];
   for (i=0;i<nbins;i++){
	 hist[i] = 0;
	 hist_private[i] = 0;
   }

   #pragma omp for nowait
   for (i = 0; i < nelts; i++){
  	 hist_private[data[i]] ++;
   }
    
   #pragma omp critical 
   for (i =0;i<nbins;i++){
	hist[i] += hist_private[i];
  }
  }//}
}
