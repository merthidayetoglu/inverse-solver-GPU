//Integration Routines
//Mert Hidayetoglu, Oct 2015
#include "vars.h"

extern double k0;

complex<double> hn(int order, double dist){
  return complex<double>(jn(order,dist),yn(order,dist));
}

static double integ(double x, double y){
  return -3*x*y+x*y*log(x*x+y*y)+x*x*atan(y/x)+y*y*atan(x/y);
}

complex<double> integrate(complex<double> post, complex<double> posb){

  extern double res;
  complex<double>numer(0,0);
  complex<double>anal(0,0);
  //NUMERICAL PART
  double dist = abs(post-posb);
  numer = j0(k0*dist);
  if(dist < res/8){
    numer = numer + complex<double>(0,0.5772156649015329*2/M_PI+2/M_PI*log(0.5));
  //else
  //  numer = numer + complex<double>(0,y0(2*M_PI*dist)-2/M_PI*log(2*M_PI*dist));
    //ANALYTICAL PART
    double xcen=(posb-post).real();
    double ycen=(posb-post).imag();
    double xmin=xcen-res/2;
    double xmax=xcen+res/2;
    double ymin=ycen-res/2;
    double ymax=ycen+res/2;
    double analt=integ(xmax,ymax)-integ(xmin,ymax)-integ(xmax,ymin)+integ(xmin,ymin);
    anal=complex<double>(0,(analt/2+log(k0)*res*res)*2/M_PI);
  }
  else
    numer =  numer + complex<double>(0,y0(k0*dist));
  numer = numer*res*res;
  return (numer+anal)*complex<double>(0,0.25);
}

complex<double> integrate_multi(complex<double> center, complex<double> posb, int order){
  extern double res;
  extern int *numsamp;
  extern int level;
  int numang = numsamp[level-1];
  double angle = 2*M_PI*order/numang;
  complex<double> u = complex<double>(cos(angle),sin(angle));
  complex<double>numer(0,0);
  //NUMERICAL PART
  complex<double> c2s = posb-center;
  numer = exp(complex<double>(0,-k0*(u.real()*c2s.real()+u.imag()*c2s.imag())));
  return numer*res*res;
}

complex<double> integrate_local(complex<double> center, complex<double> post, int order){
  extern double res;
  extern int *numsamp;
  extern int level;
  int numang = numsamp[level-1];
  double angle = 2*M_PI*order/numang;
  complex<double> u = complex<double>(cos(angle),sin(angle));
  complex<double>numer(0,0);
  //NUMERICAL PART
  complex<double> c2t = post-center;
  numer = exp(complex<double>(0,k0*(u.real()*c2t.real()+u.imag()*c2t.imag())));
  return numer;
}
