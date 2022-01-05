/**
 * Simulation over time of the N-Body problem
 * We don't pretend to follow Physics laws,
 * just some computation workload to play with.
 *
 * Based on the Basic N-Body solver from:
 * An introduction to parallel programming, 2nd Edition
 * Peter Pacheco, Matthew Malensek
 *
 * Vitor Duarte FCT/UNL 2021
 * CAD - 2021/2022
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define N_BODIES    2048
//#define N_BODIES 17		// good for development and testing
#define Gconst  0.1

double mas[N_BODIES];
double posx[N_BODIES];  /* just a 2D simulation */
double posy[N_BODIES];
double velx[N_BODIES];
double vely[N_BODIES];
double forcex[N_BODIES];
double forcey[N_BODIES];

void initParticles() {
    for (int p=0; p<N_BODIES; p++) {
        mas[p]=1;                 /* 1 mass unit */
        posx[p]=drand48()*100;    /* 100x100 space */
        posy[p]=drand48()*100;
        velx[p]=0;
        vely[p]=0;
    }
}

void printParticles(FILE *f,int rank, int n) {
    // print particles to text file f
    fprintf(f,    "#  pos     vel\n");
    for (int p=0; p<n; p++) {
        fprintf(f,"%d (%g,%g) (%g,%g) %d\n",
               n*rank +p, posx[n*rank +p], posy[n*rank +p], velx[p], vely[p],rank);
    }
}
void printParticles0(FILE *f) {
    // print particles to text file f
    fprintf(f,    "#  pos     vel\n");
    for (int p=0; p<N_BODIES; p++) {
        fprintf(f,"%d (%g,%g) (%g,%g)\n",
               p, posx[p], posy[p], velx[p], vely[p]);
    }
}


void computeForces(int q) {
    // based on the basic solver from book (not the reduced solver)
    for (int k=0; k<N_BODIES; k++) {
        if (k == q) continue; // ignore itself
        double xdiff=posx[q] - posx[k];
        double ydiff=posy[q] - posy[k];
        double dist=sqrt(xdiff*xdiff+ydiff*ydiff);
        double distCub=dist*dist*dist;
        forcex[q] -= Gconst*mas[q]*mas[k]/distCub * xdiff;
        forcey[q] -= Gconst*mas[q]*mas[k]/distCub * ydiff;
    }

}

void moveParticle(int q, double deltat,int rank, int size) {
    int n = N_BODIES/size;
    posx[n*rank +q] += deltat*velx[q];
    posy[n*rank +q] += deltat*vely[q];
    
    velx[q] += deltat/mas[n*rank +q] * forcex[n*rank +q];
    vely[q] += deltat/mas[n*rank +q] * forcey[n*rank +q];
}

void simulateStep(double deltat,int rank, int size,int n) {
    //int n = N_BODIES/size;
    memset(forcex, 0, sizeof forcex);
    memset(forcey, 0, sizeof forcey);
    if(N_BODIES != size *n && rank == size -1 ){
        for ( int q=rank*n ; q<N_BODIES; q++)
            computeForces(q);
        int rest = N_BODIES -  size *n;
        for (int q=0 ; q<n + rest; q++)
            moveParticle(q, deltat,rank, size);
    }
    else
    {
       for ( int q=rank*n ; q<rank*n + n; q++)
            computeForces(q);
        
        for (int q=0 ; q<n; q++)
            moveParticle(q, deltat,rank, size);
    }
    
    
}


int main(int argc, char *argv[]) {
    
    
    int nSteps = 100;  // default (you can give this at the command line)
    double time = 100;

    if (argc==2) {
        nSteps = atoi(argv[1]);		// number of steps
    }
    double deltat = time/nSteps;

    int size;
    int rank;
    clock_t t = clock();
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   
    int n = N_BODIES/size;
    velx[n];
    vely[n];
    
    if( rank == 0){
        initParticles();
         printf("Started %d steps!\n", nSteps);
        //printParticles0(stdout);
        
    }
    
    MPI_Bcast(mas,N_BODIES,MPI_DOUBLE,  0 ,MPI_COMM_WORLD) ;
    MPI_Bcast(posx,N_BODIES,MPI_DOUBLE,  0 ,MPI_COMM_WORLD) ;
    MPI_Bcast(posy,N_BODIES,MPI_DOUBLE,  0 ,MPI_COMM_WORLD) ;
    MPI_Scatter(velx, n, MPI_DOUBLE,velx, n, MPI_DOUBLE, 0,MPI_COMM_WORLD);
    MPI_Scatter(vely, n, MPI_DOUBLE,vely, n, MPI_DOUBLE, 0,MPI_COMM_WORLD);
    
    
    
    
    
   for (int s=0; s< nSteps; s++){
        
        simulateStep(deltat,rank,size,n);
       
        MPI_Allgather(&posx[rank*n], n, MPI_DOUBLE, posx, n, MPI_DOUBLE,MPI_COMM_WORLD);
        MPI_Allgather(&posy[rank*n], n, MPI_DOUBLE, posy, n, MPI_DOUBLE,MPI_COMM_WORLD);
        if( N_BODIES != size *n ){
            int rest = N_BODIES -  size *n;
            MPI_Bcast(&posx[size*n],rest,MPI_DOUBLE,  size -1 ,MPI_COMM_WORLD) ;
            MPI_Bcast(&posy[size*n],rest,MPI_DOUBLE,  size -1 ,MPI_COMM_WORLD) ;
            
        }
    }  
    
    double sub_velx[N_BODIES] ;
    double sub_vely[N_BODIES] ;
    

    MPI_Gather(velx, n, MPI_DOUBLE, sub_velx, n, MPI_DOUBLE, 0,MPI_COMM_WORLD);
    MPI_Gather(vely, n, MPI_DOUBLE, sub_vely, n, MPI_DOUBLE, 0,MPI_COMM_WORLD);
    int rest = N_BODIES -  size *n;
    if(N_BODIES != size *n)
        if (rank == size -1) {
            
            MPI_Send(&velx[n], rest, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&vely[n], rest, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            
        } else if (rank == 0) {
            MPI_Recv(&sub_velx[size*n], rest, MPI_DOUBLE, size -1, 0, MPI_COMM_WORLD ,MPI_STATUS_IGNORE);
            MPI_Recv(&sub_vely[size*n], rest, MPI_DOUBLE, size -1, 0, MPI_COMM_WORLD ,MPI_STATUS_IGNORE);
        }

    /*if (rank == 0) {      
        fprintf(stdout,    "#  pos     vel\n");
        for (int p=0; p<N_BODIES; p++) {
            fprintf(stdout,"%d (%g,%g) (%g,%g)\n",
                p, posx[p], posy[p], sub_velx[p], sub_vely[p]);
        }
        
    }*/
    t = clock()-t;
    //printParticles(stdout);		// check if this solution is correct
    if (rank == 0)
        printf("time: %f s\n", t/(double)CLOCKS_PER_SEC);
    
    MPI_Finalize();
    return 0;
}
