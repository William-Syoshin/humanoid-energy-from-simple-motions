/* Reconstructed Webots contact instrumentation. Forces are ODE contact reactions,
   not inferred from contact counts. Explicit friction/contact parameters are
   recorded; this replaces external contacts only. Internal collisions unchanged. */
#include <plugins/physics.h>
#include <math.h>
#include <string.h>
#define CAP 4096
static dGeomID floor_geom, object_geom;
static dBodyID left_foot, right_foot, object_body;
static dJointFeedback feedback[2][CAP];
static double normals[2][CAP][3];
static int kinds[2][CAP], counts[2], write_buffer=0, read_buffer=1;
static int debug_count=0;
static pthread_mutex_t mutex;
void webots_physics_init() {
 pthread_mutex_init(&mutex, NULL);
 floor_geom=dWebotsGetGeomFromDEF("FLOOR");
 object_geom=dWebotsGetGeomFromDEF("OBJECT");
 object_body=dWebotsGetBodyFromDEF("OBJECT");
 left_foot=dWebotsGetBodyFromDEF("BODY_LeftFoot");
 right_foot=dWebotsGetBodyFromDEF("BODY_RightFoot");
 dWebotsConsolePrintf("CONTACT_INIT floor=%p object=%p left=%p right=%p\n",floor_geom,object_geom,left_foot,right_foot);
}
void webots_physics_step() { read_buffer=write_buffer;write_buffer=1-write_buffer;counts[write_buffer]=0; }
int webots_physics_collide(dGeomID g1,dGeomID g2) {
 int floor=(g1==floor_geom||g2==floor_geom);
 int object=(g1==object_geom||g2==object_geom);
 if(!floor&&!object) return 0;
 dBodyID b1=dGeomGetBody(g1), b2=dGeomGetBody(g2);
 dBodyID robot_body=(g1==floor_geom||g1==object_geom)?b2:b1;
 if(!robot_body||robot_body==object_body) return 0;
 dContact c[16]; memset(c,0,sizeof(c));
 int n=dCollide(g1,g2,16,&c[0].geom,sizeof(dContact));
 pthread_mutex_lock(&mutex);
 int nfeedback=counts[write_buffer];
 for(int i=0;i<n&&nfeedback<CAP;i++) {
  c[i].surface.mode=dContactApprox1|dContactSoftERP|dContactSoftCFM;
  c[i].surface.mu=object?0.4:1.0;
  c[i].surface.soft_erp=0.2; c[i].surface.soft_cfm=1e-5;
  dJointID j=dJointCreateContact(dBodyGetWorld(robot_body),dWebotsGetContactJointGroup(),&c[i]);
  dJointAttach(j,b1,b2);
  memset(&feedback[write_buffer][nfeedback],0,sizeof(dJointFeedback));
  dJointSetFeedback(j,&feedback[write_buffer][nfeedback]);
  for(int k=0;k<3;k++) normals[write_buffer][nfeedback][k]=c[i].geom.normal[k];
  kinds[write_buffer][nfeedback]=object?3:(robot_body==left_foot?1:(robot_body==right_foot?2:0));
  nfeedback++;
 }
 counts[write_buffer]=nfeedback;
 pthread_mutex_unlock(&mutex);
 return 1;
}
void webots_physics_step_end() {
 /* Fn,Ft,Fnonfoot,Fobject,leftFn,rightFn,contacts. No labels or oracle in plugin. */
 double v[7]={0}; int nfeedback=counts[read_buffer];
 for(int i=0;i<nfeedback;i++) {
  const dReal *f=feedback[read_buffer][i].f1;
  double n1=0,n2=0; for(int k=0;k<3;k++){n1+=f[k]*f[k];n2+=feedback[read_buffer][i].f2[k]*feedback[read_buffer][i].f2[k];} if(n2>n1) f=feedback[read_buffer][i].f2;
  double norm2=0,proj=0;
  for(int k=0;k<3;k++){ norm2+=f[k]*f[k];proj+=f[k]*normals[read_buffer][i][k]; }
  double normal=fabs(proj),tangent=sqrt(fmax(0,norm2-proj*proj));
  v[0]+=normal;v[1]+=tangent;
  if(kinds[read_buffer][i]==0) v[2]+=sqrt(norm2);
  if(kinds[read_buffer][i]==3) v[3]+=sqrt(norm2);
  if(kinds[read_buffer][i]==1) v[4]+=normal;
  if(kinds[read_buffer][i]==2) v[5]+=normal;
 }
 if(nfeedback>0 && debug_count++<6) dWebotsConsolePrintf("FORCE_DEBUG t=%g n=%d force=%g,%g,%g f2=%g,%g,%g\n", dWebotsGetTime(), nfeedback,feedback[read_buffer][0].f1[0],feedback[read_buffer][0].f1[1],feedback[read_buffer][0].f1[2],feedback[read_buffer][0].f2[0],feedback[read_buffer][0].f2[1],feedback[read_buffer][0].f2[2]);
 v[6]=nfeedback;dWebotsSend(7,v,sizeof(v)); counts[read_buffer]=0;
}
void webots_physics_cleanup(){pthread_mutex_destroy(&mutex);}
