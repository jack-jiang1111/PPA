void merge(a,b,lo,mid,hi)
int a[],b[], lo,mid,hi;
{
  int h,i,j,k;
  h = lo;
  i = lo;
  j = mid+1;
  while ((h<=mid) && (j<=hi)){
    if (a[h]<=a[j]) {
      b[i++] = a[h++]; 
    }
    else {
      b[i++] = a[j++]; 
    }
  }

  if (h>mid){
    for(k=j;k<=hi;k++){
      b[i++] = a[k];
    }  
  }
  else{  
    for(k=h;k<=mid;k++){
      b[i++] = a[k];
    }  
  }

  for(k=lo;k<=hi;k++){
    a[k] = b[k];
  } 
}


void msort_seq(int a[],int b[],int lo, int hi)
{
  int temp,mid;
  if (lo < hi){ 
    if (hi == lo+1){
      if (a[hi]<a[lo]) {
        temp=a[hi];a[hi]=a[lo];a[lo]=temp;
      }
    }
    else{
      mid = (lo+hi)/2;
      msort_seq(a,b,lo,mid);
      msort_seq(a,b,mid+1,hi);
      merge(a,b,lo,mid,hi);
    }
  }
}

