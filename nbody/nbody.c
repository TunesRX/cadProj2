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

#define N_BODIES    128
//#define N_BODIES 16		// good for development and testing
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

void printParticles(FILE *f) {
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

void moveParticle(int q, double deltat) {
    posx[q] += deltat*velx[q];
    posy[q] += deltat*vely[q];
    velx[q] += deltat/mas[q] * forcex[q];
    vely[q] += deltat/mas[q] * forcey[q];
}

void simulateStep(double deltat) {
    memset(forcex, 0, sizeof forcex);
    memset(forcey, 0, sizeof forcey);
    for ( int q=0; q<N_BODIES; q++)
        computeForces(q);
    for (int q=0; q<N_BODIES; q++)
        moveParticle(q, deltat);
}


int main(int argc, char *argv[]) {
    int nSteps = 100;  // default (you can give this at the command line)
    double time = 100;

    if (argc==2) {
        nSteps = atoi(argv[1]);		// number of steps
    }
    double deltat = time/nSteps;

    printf("Started %d steps!\n", nSteps);

    initParticles();
    clock_t t = clock();

    for (int s=0; s< nSteps; s++)
        simulateStep(deltat);

    t = clock()-t;
    printParticles(stdout);		// check if this solution is correct
    //printf("time: %f s\n", t/(double)CLOCKS_PER_SEC);

    return 0;
}