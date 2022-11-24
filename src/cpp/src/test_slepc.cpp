
#include "petsc.h"
#include "petscksp.h"
#include "petscviewerhdf5.h"
#include <iostream>
#include <math.h>
#include <cstdlib>
#include "slepc.h"
#include "slepcbv.h"

// static PetscErrorCode read_matrix(Mat &M, char *name,  PetscViewer fd);
// PetscErrorCode solve_ksps(const Mat &M, const Mat &C, Mat &Y);


int main( int argc, char *argv[]) {

    Mat             M, Mk;
    Mat             OUT;
    Vec             v, tmp_column_vector, tmp_row_vector;

    PetscMPIInt     rank, size;
    BV              Q;
    PetscInt        ncols=10, nrows=10;

    PetscInt        nvects = 5;
    PetscBool       isdef;
    PetscErrorCode  ierr;
    PetscRandom      rctx;


	SlepcInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    MPI_Comm_size(PETSC_COMM_WORLD,&size);

    // init rgn
    PetscRandomCreate(PETSC_COMM_WORLD,&rctx);

    // create random matrix M
    MatCreate(PETSC_COMM_WORLD, &M);
    MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,nrows, ncols);
    MatSetFromOptions(M);
    MatSetUp(M);
    MatSetRandom(M,rctx);


    // init random BV Q
    BVCreate(PETSC_COMM_WORLD, &Q);
    BVSetSizes(Q, PETSC_DECIDE, nrows, nvects);
    BVSetFromOptions(Q);
    BVSetRandom(Q);
    // BVView(Q, PETSC_VIEWER_STDOUT_WORLD);

    // create random matrix OUT
    MatCreate(PETSC_COMM_WORLD, &OUT);
    MatSetSizes(OUT,PETSC_DECIDE,PETSC_DECIDE,nrows, nvects);
    MatSetFromOptions(OUT);
    MatSetUp(OUT);

    // create random vectors v
    VecCreate(PETSC_COMM_WORLD, &v);
    VecSetSizes(v, PETSC_DECIDE, nrows);
    VecSetFromOptions(v);
    VecSetRandom(v, rctx);

    // create null vector
    VecCreate(PETSC_COMM_WORLD, &tmp_column_vector);
    VecSetSizes(tmp_column_vector, PETSC_DECIDE, nrows);
    VecSetFromOptions(tmp_column_vector);

    // compute tmp_vect = M v
    MatMult(M, v, tmp_column_vector);

    // compute tmp_row_vect = Q.T M v
    VecCreate(PETSC_COMM_WORLD, &tmp_row_vector);
    VecSetSizes(tmp_row_vector,PETSC_DECIDE,ncols);
    VecSetFromOptions(tmp_row_vector);
    BVDotVec(Q, tmp_column_vector, (PetscScalar *)tmp_row_vector);
    // VecView((Vec)tmp_row_vector, PETSC_VIEWER_STDOUT_WORLD);

    // v = v - Q Q.T M v
    BVMultVec(Q, -1.0, 1.0, v, (PetscScalar *)tmp_row_vector);
    VecView(v, PETSC_VIEWER_STDOUT_WORLD);

    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nvects, nvects, NULL, &Mk);
    // MatSetSizes(Mk,PETSC_DECIDE,PETSC_DECIDE,nvects, nvects);
    // MatSetFromOptions(Mk);
    // MatSetUp(Mk);
    BVMatProject(Q, M, Q, Mk);

	PetscFinalize();
	return 0;
}



