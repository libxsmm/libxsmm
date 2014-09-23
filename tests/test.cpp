#include<stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <micsmm.h>

__declspec( target (mic))
double mytime() {
  timeval a;
  gettimeofday(&a, 0);
  return (double)(a.tv_sec*1000 + a.tv_usec/1000.0);
}

__declspec( target (mic))
void bench_4(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 7812500;
double* a = new double[16];
double* b = new double[16];
double* c = new double[16];

for(int i=0;i<16;i++){
a[i]=((double)i)/((double)16);
b[i]=((double)i)/((double)16);
c[i]=((double)i)/((double)16);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_4_4_4(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)128)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<16;i++){
a[i]=((double)i)/((double)16);
b[i]=((double)i)/((double)16);
c[i]=((double)i)/((double)16);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_4_4_4(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)128)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 4, perf1, perf2);
}
__declspec( target (mic))
void bench_5(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 4000000;
double* a = new double[25];
double* b = new double[25];
double* c = new double[25];

for(int i=0;i<25;i++){
a[i]=((double)i)/((double)25);
b[i]=((double)i)/((double)25);
c[i]=((double)i)/((double)25);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_5_5_5(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)250)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<25;i++){
a[i]=((double)i)/((double)25);
b[i]=((double)i)/((double)25);
c[i]=((double)i)/((double)25);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_5_5_5(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)250)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 5, perf1, perf2);
}
__declspec( target (mic))
void bench_6(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 2314814;
double* a = new double[36];
double* b = new double[36];
double* c = new double[36];

for(int i=0;i<36;i++){
a[i]=((double)i)/((double)36);
b[i]=((double)i)/((double)36);
c[i]=((double)i)/((double)36);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_6_6_6(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)432)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<36;i++){
a[i]=((double)i)/((double)36);
b[i]=((double)i)/((double)36);
c[i]=((double)i)/((double)36);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_6_6_6(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)432)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 6, perf1, perf2);
}
__declspec( target (mic))
void bench_7(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 1457725;
double* a = new double[49];
double* b = new double[49];
double* c = new double[49];

for(int i=0;i<49;i++){
a[i]=((double)i)/((double)49);
b[i]=((double)i)/((double)49);
c[i]=((double)i)/((double)49);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_7_7_7(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)686)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<49;i++){
a[i]=((double)i)/((double)49);
b[i]=((double)i)/((double)49);
c[i]=((double)i)/((double)49);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_7_7_7(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)686)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 7, perf1, perf2);
}
__declspec( target (mic))
void bench_8(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 976562;
double* a = new double[64];
double* b = new double[64];
double* c = new double[64];

for(int i=0;i<64;i++){
a[i]=((double)i)/((double)64);
b[i]=((double)i)/((double)64);
c[i]=((double)i)/((double)64);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_8_8_8(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)1024)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<64;i++){
a[i]=((double)i)/((double)64);
b[i]=((double)i)/((double)64);
c[i]=((double)i)/((double)64);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_8_8_8(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)1024)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 8, perf1, perf2);
}
__declspec( target (mic))
void bench_9(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 685871;
double* a = new double[81];
double* b = new double[81];
double* c = new double[81];

for(int i=0;i<81;i++){
a[i]=((double)i)/((double)81);
b[i]=((double)i)/((double)81);
c[i]=((double)i)/((double)81);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_9_9_9(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)1458)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<81;i++){
a[i]=((double)i)/((double)81);
b[i]=((double)i)/((double)81);
c[i]=((double)i)/((double)81);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_9_9_9(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)1458)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 9, perf1, perf2);
}
__declspec( target (mic))
void bench_10(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 500000;
double* a = new double[100];
double* b = new double[100];
double* c = new double[100];

for(int i=0;i<100;i++){
a[i]=((double)i)/((double)100);
b[i]=((double)i)/((double)100);
c[i]=((double)i)/((double)100);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_10_10_10(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)2000)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<100;i++){
a[i]=((double)i)/((double)100);
b[i]=((double)i)/((double)100);
c[i]=((double)i)/((double)100);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_10_10_10(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)2000)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 10, perf1, perf2);
}
__declspec( target (mic))
void bench_11(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 375657;
double* a = new double[121];
double* b = new double[121];
double* c = new double[121];

for(int i=0;i<121;i++){
a[i]=((double)i)/((double)121);
b[i]=((double)i)/((double)121);
c[i]=((double)i)/((double)121);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_11_11_11(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)2662)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<121;i++){
a[i]=((double)i)/((double)121);
b[i]=((double)i)/((double)121);
c[i]=((double)i)/((double)121);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_11_11_11(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)2662)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 11, perf1, perf2);
}
__declspec( target (mic))
void bench_12(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 289351;
double* a = new double[144];
double* b = new double[144];
double* c = new double[144];

for(int i=0;i<144;i++){
a[i]=((double)i)/((double)144);
b[i]=((double)i)/((double)144);
c[i]=((double)i)/((double)144);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_12_12_12(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)3456)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<144;i++){
a[i]=((double)i)/((double)144);
b[i]=((double)i)/((double)144);
c[i]=((double)i)/((double)144);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_12_12_12(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)3456)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 12, perf1, perf2);
}
__declspec( target (mic))
void bench_13(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 227583;
double* a = new double[169];
double* b = new double[169];
double* c = new double[169];

for(int i=0;i<169;i++){
a[i]=((double)i)/((double)169);
b[i]=((double)i)/((double)169);
c[i]=((double)i)/((double)169);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_13_13_13(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)4394)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<169;i++){
a[i]=((double)i)/((double)169);
b[i]=((double)i)/((double)169);
c[i]=((double)i)/((double)169);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_13_13_13(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)4394)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 13, perf1, perf2);
}
__declspec( target (mic))
void bench_14(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 182215;
double* a = new double[196];
double* b = new double[196];
double* c = new double[196];

for(int i=0;i<196;i++){
a[i]=((double)i)/((double)196);
b[i]=((double)i)/((double)196);
c[i]=((double)i)/((double)196);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_14_14_14(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)5488)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<196;i++){
a[i]=((double)i)/((double)196);
b[i]=((double)i)/((double)196);
c[i]=((double)i)/((double)196);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_14_14_14(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)5488)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 14, perf1, perf2);
}
__declspec( target (mic))
void bench_15(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 148148;
double* a = new double[225];
double* b = new double[225];
double* c = new double[225];

for(int i=0;i<225;i++){
a[i]=((double)i)/((double)225);
b[i]=((double)i)/((double)225);
c[i]=((double)i)/((double)225);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_15_15_15(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)6750)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<225;i++){
a[i]=((double)i)/((double)225);
b[i]=((double)i)/((double)225);
c[i]=((double)i)/((double)225);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_15_15_15(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)6750)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 15, perf1, perf2);
}
__declspec( target (mic))
void bench_16(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 122070;
double* a = new double[256];
double* b = new double[256];
double* c = new double[256];

for(int i=0;i<256;i++){
a[i]=((double)i)/((double)256);
b[i]=((double)i)/((double)256);
c[i]=((double)i)/((double)256);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_16_16_16(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)8192)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<256;i++){
a[i]=((double)i)/((double)256);
b[i]=((double)i)/((double)256);
c[i]=((double)i)/((double)256);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_16_16_16(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)8192)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 16, perf1, perf2);
}
__declspec( target (mic))
void bench_17(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 101770;
double* a = new double[289];
double* b = new double[289];
double* c = new double[289];

for(int i=0;i<289;i++){
a[i]=((double)i)/((double)289);
b[i]=((double)i)/((double)289);
c[i]=((double)i)/((double)289);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_17_17_17(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)9826)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<289;i++){
a[i]=((double)i)/((double)289);
b[i]=((double)i)/((double)289);
c[i]=((double)i)/((double)289);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_17_17_17(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)9826)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 17, perf1, perf2);
}
__declspec( target (mic))
void bench_18(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 85733;
double* a = new double[324];
double* b = new double[324];
double* c = new double[324];

for(int i=0;i<324;i++){
a[i]=((double)i)/((double)324);
b[i]=((double)i)/((double)324);
c[i]=((double)i)/((double)324);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_18_18_18(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)11664)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<324;i++){
a[i]=((double)i)/((double)324);
b[i]=((double)i)/((double)324);
c[i]=((double)i)/((double)324);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_18_18_18(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)11664)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 18, perf1, perf2);
}
__declspec( target (mic))
void bench_19(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 72896;
double* a = new double[361];
double* b = new double[361];
double* c = new double[361];

for(int i=0;i<361;i++){
a[i]=((double)i)/((double)361);
b[i]=((double)i)/((double)361);
c[i]=((double)i)/((double)361);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_19_19_19(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)13718)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<361;i++){
a[i]=((double)i)/((double)361);
b[i]=((double)i)/((double)361);
c[i]=((double)i)/((double)361);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_19_19_19(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)13718)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 19, perf1, perf2);
}
__declspec( target (mic))
void bench_20(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 62500;
double* a = new double[400];
double* b = new double[400];
double* c = new double[400];

for(int i=0;i<400;i++){
a[i]=((double)i)/((double)400);
b[i]=((double)i)/((double)400);
c[i]=((double)i)/((double)400);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_20_20_20(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)16000)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<400;i++){
a[i]=((double)i)/((double)400);
b[i]=((double)i)/((double)400);
c[i]=((double)i)/((double)400);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_20_20_20(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)16000)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 20, perf1, perf2);
}
__declspec( target (mic))
void bench_21(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 53989;
double* a = new double[441];
double* b = new double[441];
double* c = new double[441];

for(int i=0;i<441;i++){
a[i]=((double)i)/((double)441);
b[i]=((double)i)/((double)441);
c[i]=((double)i)/((double)441);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_21_21_21(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)18522)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<441;i++){
a[i]=((double)i)/((double)441);
b[i]=((double)i)/((double)441);
c[i]=((double)i)/((double)441);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_21_21_21(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)18522)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 21, perf1, perf2);
}
__declspec( target (mic))
void bench_22(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 46957;
double* a = new double[484];
double* b = new double[484];
double* c = new double[484];

for(int i=0;i<484;i++){
a[i]=((double)i)/((double)484);
b[i]=((double)i)/((double)484);
c[i]=((double)i)/((double)484);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_22_22_22(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)21296)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<484;i++){
a[i]=((double)i)/((double)484);
b[i]=((double)i)/((double)484);
c[i]=((double)i)/((double)484);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_22_22_22(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)21296)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 22, perf1, perf2);
}
__declspec( target (mic))
void bench_23(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 41094;
double* a = new double[529];
double* b = new double[529];
double* c = new double[529];

for(int i=0;i<529;i++){
a[i]=((double)i)/((double)529);
b[i]=((double)i)/((double)529);
c[i]=((double)i)/((double)529);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_23_23_23(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)24334)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<529;i++){
a[i]=((double)i)/((double)529);
b[i]=((double)i)/((double)529);
c[i]=((double)i)/((double)529);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_23_23_23(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)24334)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 23, perf1, perf2);
}
__declspec( target (mic))
void bench_24(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 36168;
double* a = new double[576];
double* b = new double[576];
double* c = new double[576];

for(int i=0;i<576;i++){
a[i]=((double)i)/((double)576);
b[i]=((double)i)/((double)576);
c[i]=((double)i)/((double)576);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_24_24_24(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)27648)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<576;i++){
a[i]=((double)i)/((double)576);
b[i]=((double)i)/((double)576);
c[i]=((double)i)/((double)576);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_24_24_24(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)27648)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 24, perf1, perf2);
}
__declspec( target (mic))
void bench_25(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 32000;
double* a = new double[625];
double* b = new double[625];
double* c = new double[625];

for(int i=0;i<625;i++){
a[i]=((double)i)/((double)625);
b[i]=((double)i)/((double)625);
c[i]=((double)i)/((double)625);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_25_25_25(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)31250)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<625;i++){
a[i]=((double)i)/((double)625);
b[i]=((double)i)/((double)625);
c[i]=((double)i)/((double)625);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_25_25_25(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)31250)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 25, perf1, perf2);
}
__declspec( target (mic))
void bench_26(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 28447;
double* a = new double[676];
double* b = new double[676];
double* c = new double[676];

for(int i=0;i<676;i++){
a[i]=((double)i)/((double)676);
b[i]=((double)i)/((double)676);
c[i]=((double)i)/((double)676);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_26_26_26(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)35152)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<676;i++){
a[i]=((double)i)/((double)676);
b[i]=((double)i)/((double)676);
c[i]=((double)i)/((double)676);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_26_26_26(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)35152)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 26, perf1, perf2);
}
__declspec( target (mic))
void bench_27(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 25402;
double* a = new double[729];
double* b = new double[729];
double* c = new double[729];

for(int i=0;i<729;i++){
a[i]=((double)i)/((double)729);
b[i]=((double)i)/((double)729);
c[i]=((double)i)/((double)729);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_27_27_27(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)39366)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<729;i++){
a[i]=((double)i)/((double)729);
b[i]=((double)i)/((double)729);
c[i]=((double)i)/((double)729);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_27_27_27(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)39366)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 27, perf1, perf2);
}
__declspec( target (mic))
void bench_28(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 22776;
double* a = new double[784];
double* b = new double[784];
double* c = new double[784];

for(int i=0;i<784;i++){
a[i]=((double)i)/((double)784);
b[i]=((double)i)/((double)784);
c[i]=((double)i)/((double)784);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_28_28_28(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)43904)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<784;i++){
a[i]=((double)i)/((double)784);
b[i]=((double)i)/((double)784);
c[i]=((double)i)/((double)784);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_28_28_28(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)43904)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 28, perf1, perf2);
}
__declspec( target (mic))
void bench_29(void){
int length=1;
double t1,t2;
double perf1,perf2;
long iterations = 20501;
double* a = new double[841];
double* b = new double[841];
double* c = new double[841];

for(int i=0;i<841;i++){
a[i]=((double)i)/((double)841);
b[i]=((double)i)/((double)841);
c[i]=((double)i)/((double)841);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_1_29_29_29(a,b,c);
}
t2=mytime();
perf1=((double)iterations)*((double)48778)/(t2-t1)/((double) 1000)/((double) 1000);

for(int i=0;i<841;i++){
a[i]=((double)i)/((double)841);
b[i]=((double)i)/((double)841);
c[i]=((double)i)/((double)841);
}
t1=mytime();
for(long n=0;n<iterations;n++){
micgemm_2_29_29_29(a,b,c);
}
t2=mytime();
perf2=((double)iterations)*((double)48778)/(t2-t1)/((double) 1000)/((double) 1000);

printf("%d \t%f \t %f \n", 29, perf1, perf2);
}
int main(void){
#pragma offload target(mic:0)
{
printf("starting benchmark\n");
bench_4();
bench_5();
bench_6();
bench_7();
bench_8();
bench_9();
bench_10();
bench_11();
bench_12();
bench_13();
bench_14();
bench_15();
bench_16();
bench_17();
bench_18();
bench_19();
bench_20();
bench_21();
bench_22();
bench_23();
bench_24();
bench_25();
bench_26();
bench_27();
bench_28();
bench_29();
}
}
