#include <iostream>
#include <cmath>
#include <complex>

#include "petsc.h"
#include "slepc.h"

#include "doehler_eigensolver_kernel.hpp"

// Run parameters:
// ./doehler_eigensolver -if ../python/A_M_T_matrices.dat -A_name A -M_name M -T_name T -data_format binary -of ./output_file.mat -show_input FALSE

namespace doehler{
  void eigen_doehler_petsc(Mat &A_Mat, Mat &M_Mat, Mat &T_Mat_in, PetscInt n_eigs, PetscInt n_max_iter, 
      PetscReal tol, bool normalize_S, bool verbose, 
      bool indefinite_dot, bool py_inputs, MPI_Comm comm)
  {
    std::cout << "\n**************************************************************" << std::endl;
    std::cout << "* Doehler eingenvalue solver (PETSc) START" << std::endl;
    std::cout << "**************************************************************\n" << std::endl;
    
    // Initialize the algorithm with initial guesses
    
    // Compute initial parameters, like system size, etc 
    PetscInt A_n_rows, A_n_cols;  // number of rows and columns of matrix A
    MatGetSize(A_Mat, &A_n_rows, &A_n_cols);
    
    std::cout << "System size:" << std::endl;
    std::cout << "   n_rows: " << A_n_rows << std::endl;
    std::cout << "   n_cols: " << A_n_cols << std::endl;
    
    // Initialize tranformation matrix T as a bv system (used to project to the reduced space)
    BV T_bv;
    BVCreate(comm, &T_bv);
    
    BVSetSizes(T_bv, PETSC_DECIDE, A_n_rows, 2*n_eigs);
    BVSetFromOptions(T_bv);
    
    // Initialize eigenvector bv system to store the eigenvectors computed each iteration
    BV Q_bv;
    BVCreate(comm, &Q_bv);
    
    BVSetSizes(Q_bv, PETSC_DECIDE, 2*n_eigs, 2*n_eigs);
    BVSetFromOptions(Q_bv);
    
    // Set the initial (guess) values
    // (the n_eigs eigenvectors we wish to find and the 
    // n_eigs vector search directions)
    if(py_inputs)
    {
      // Use initial values from python for one to one comparison
      Mat T_Mat;
      MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, A_n_rows, 2*n_eigs, NULL, &T_Mat);
      MatSetUp(T_Mat); 
      BVGetMat(T_bv, &T_Mat);
      MatCopy(T_Mat_in, T_Mat, DIFFERENT_NONZERO_PATTERN);
      BVRestoreMat(T_bv, &T_Mat);
    }
    else
    {
      // Use randomly generated values to initialize T_bv
      BVSetRandom(T_bv);
    }
    
    // TODO Apply the projector to ensure X and S satisfy the constraint
    // TODO Consider forcing all small matrices and vectors to be sequential
    
    // Iterate to find corrected solutions to the eigenvalue
    PetscInt iter_idx = 0;  // initialize counter for number of interations performed
    PetscReal error_max = 1.0;  // initialize error max to determine if the loop is over or not
    
    // Declare variables required in while loop 
    
    // Reduced matrices obtained by projecting A_Mat and M_Mat
    // into the T_bv space (approximate eigenvectors \ocirc search space) 
    Mat A_Mat_p, M_Mat_p;
    MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, 2*n_eigs, 2*n_eigs, NULL, &A_Mat_p);
    MatSetUp(A_Mat_p); 
    MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, 2*n_eigs, 2*n_eigs, NULL, &M_Mat_p);
    MatSetUp(M_Mat_p); 
                           
    // Vector containing eigenvalues 
    Vec L_Vec;
    
    // Setup the (small) eigensolver 
    EPS eigen_solver;
    EPSCreate(comm, &eigen_solver);
    
    do
    {
      iter_idx++;  // update counter for number of iterations performed
      
      // Compute the reduced matrices on the space spanned by T = [X, S]
      BVMatProject(T_bv, A_Mat, T_bv, A_Mat_p);
      BVMatProject(T_bv, M_Mat, T_bv, M_Mat_p);
      
      // std::cout << "\n\nThe A matrix:" << std::endl;
      // MatView(A_Mat_p, PETSC_VIEWER_STDOUT_WORLD);

      // std::cout << "\n\nThe M matrix:" << std::endl;
      // MatView(M_Mat_p, PETSC_VIEWER_STDOUT_WORLD);
      
      // Create temporary vector to store eigenvalues
      MatCreateVecs(M_Mat_p, &L_Vec, NULL);
      
      // std::cout << "\n\nThe L_Vec vector:" << std::endl;
      // VecView(L_Vec, PETSC_VIEWER_STDOUT_WORLD);
      
      // Make sure the resulting reduced matrices are still symmetric
      // Symmetry can be lost due to roundoff and accumulation errors
      
      // Compute the Ritz values (L) and Ritz vectors (Q) of the reduced eigenvalue problem
      EPSSetOperators(eigen_solver, A_Mat_p, M_Mat_p);
      EPSSetProblemType(eigen_solver, EPS_GHEP);
      EPSSetWhichEigenpairs(eigen_solver, EPS_SMALLEST_REAL);
      EPSSetFromOptions(eigen_solver);
      EPSSolve(eigen_solver);
      
      // Save all eigenvectors into the BV Q_bv
      // and normalise the eigenvectors using M_Mat_p as the inner product matrix
      if(!indefinite_dot)  // need to check if bool is correct
        BVSetMatrix(Q_bv, M_Mat_p, PETSC_FALSE);
      
      for(PetscInt eigen_v_idx = 0; eigen_v_idx < 2*n_eigs; eigen_v_idx++)
      {
        Vec eigen_v;  // temporary eigenvector to extract from EPS and store in Q_bv
        PetscScalar L_value;  // temporary eigenvalue to extract from EPS and store in L_Vec
        
        // Extract the eigenvector and eigenvalue
        BVGetColumn(Q_bv, eigen_v_idx, &eigen_v);  // first get the column to update 
        EPSGetEigenpair(eigen_solver, eigen_v_idx, &L_value, NULL, eigen_v, NULL);  // update the column with the eigenvector
        VecSetValue(L_Vec, eigen_v_idx, L_value, INSERT_VALUES);  // update the eigenvalue
        
        if(indefinite_dot)
        {
          // Normalise the eigenector using just the transpose, without conjugation
          Vec M_mult_eigen_v;
          VecDuplicate(eigen_v, &M_mult_eigen_v);  // create a temporary vector with the same shape for storing M_Mat_p * eigen_v
          MatMult(M_Mat_p, eigen_v, M_mult_eigen_v);  // compute M_Mat_p * eigen_v
          PetscScalar eigen_v_norm_squared;  // the square of the norm of the eigenvector, here is potentially complex since it is a pseudo-norm
          VecTDot(M_mult_eigen_v, eigen_v, &eigen_v_norm_squared);  // compute the indefinite vector dot product squared
          VecScale(eigen_v, 1.0/std::sqrt(eigen_v_norm_squared));  // normalise it
                    
          BVRestoreColumn(Q_bv, eigen_v_idx, &eigen_v);  // restore the column so that we can reuse Q_bv
        }
        else 
        {
          // Return the updated column vector to Q_bv
          BVRestoreColumn(Q_bv, eigen_v_idx, &eigen_v);  // restore the column so that we can reuse Q_bv

          // Normalise it using BV (this uses the conjugate transpose)
          PetscReal eigen_v_norm;  // note that here it need to be a real since here it is a real norm
          BVNormColumn(Q_bv, eigen_v_idx, NORM_2, &eigen_v_norm);  // compute the norm of the eigenvector
          BVScaleColumn(Q_bv, eigen_v_idx, eigen_v_norm);  // normalise it 
          
          // std::cout << "\n\nEigenvector:"  << eigen_v_idx << std::endl;
          // VecView(eigen_v, PETSC_VIEWER_STDOUT_WORLD);
        }
        
        // BVGetColumn(Q_bv, eigen_v_idx, &eigen_v); 
        // std::cout << "\n\nEigenvector:"  << eigen_v_idx << std::endl;
        // VecView(eigen_v, PETSC_VIEWER_STDOUT_WORLD);
        // BVRestoreColumn(Q_bv, eigen_v_idx, &eigen_v);  // restore the column so that we can reuse Q_bv        
      }
      
      // std::cout << "\n\nEigenvalues:" << std::endl;
      // VecView(L_Vec, PETSC_VIEWER_STDOUT_WORLD);
      
      // Now we can reconstruct the eigenvectors, i.e., compute them in the full space
      BV T_bv_new;  // the updated reconstructed vectors, below just make a copy to start with
      BVDuplicate(T_bv, &T_bv_new);
      BVCopy(T_bv, T_bv_new);
      
      // Reconstruct the eigenvectors (all at once)
      Mat Q_Mat;  // get the matrix associated to the search space eigenvectors to use with mult below
      BVGetMat(Q_bv, &Q_Mat); 
      BVMultInPlace(T_bv_new, Q_Mat, 0, 2*n_eigs); // make the multiplication T_bv_new = T_bv * Q_mat (reconstruct eigenvalues)  
      BVRestoreMat(Q_bv, &Q_Mat);  // restore the matrix so that we can reuse Q_bv
      
      BV X_bv;  // the approximate solution of eigenvalues 
      BVCreate(comm, &X_bv);
      BVSetSizes(X_bv, PETSC_DECIDE, A_n_rows, n_eigs);
      BVSetFromOptions(X_bv);
      
      BVSetActiveColumns(T_bv_new, 0, n_eigs);  // activate the columns associated to the approximate solution
      BVCopy(T_bv_new, X_bv);
      BVSetActiveColumns(T_bv_new, 0, 2*n_eigs);  // always return to original state
      
      BV W_r_bv;  // the search space
      BVCreate(comm, &W_r_bv);
      BVSetSizes(W_r_bv, PETSC_DECIDE, A_n_rows, n_eigs);
      BVSetFromOptions(W_r_bv);
      
      BVSetActiveColumns(T_bv_new, n_eigs, 2*n_eigs);  // activate the columns associated to the search space
      BVCopy(T_bv_new, W_r_bv);
      BVSetActiveColumns(T_bv_new, 0, 2*n_eigs);  // always return to original state
      
      // Compute the residual with the updated eigenvectors
      BV R_bv;  // the residual column vectors
      BVCreate(comm, &R_bv);
      BVSetSizes(R_bv, PETSC_DECIDE, A_n_rows, n_eigs);
      BVSetFromOptions(R_bv);
      doehler::compute_residual_eigen_v(A_Mat, M_Mat, L_Vec, X_bv, 0, n_eigs, R_bv);
       
      // Compute the max "L2" error norm
      error_max = 0.0;  // initialise the maximum error of all eigenvectors
                        // start with zero value to be sure we update it
      PetscReal error_max_temp = 0.0;  // temporary storage for maximum error used in for loop below
                                      // to compute the maximum error
      for(PetscInt eigen_v_idx = 0; eigen_v_idx < n_eigs; eigen_v_idx++)
      {
        BVNormColumn(R_bv,eigen_v_idx, NORM_2, &error_max_temp);
        error_max = (error_max_temp > error_max) ? error_max_temp : error_max;
      }
      
      if(verbose)
        std::cout << "iter " << iter_idx << ": \t error_max = " << error_max << std::endl;

      // TODO Apply preconditioner  
      
      // Compute the new augmented solution space (the correction space) and the new search space
      BV RR_bv;  // the BV containing the residual of the residual BV
      BVCreate(comm, &RR_bv);
      BVSetSizes(RR_bv, PETSC_DECIDE, A_n_rows, n_eigs);
      BVSetFromOptions(RR_bv);
      doehler::compute_residual_eigen_v(A_Mat, M_Mat, L_Vec, R_bv, n_eigs, n_eigs, RR_bv);
      
      BVSetActiveColumns(T_bv_new, n_eigs, 2*n_eigs);
      Mat W_r_bv_Mat;
      BVGetMat(W_r_bv, &W_r_bv_Mat);
      
      BV V_bv;
      BVCreate(comm, &V_bv);
      BVSetSizes(V_bv, PETSC_DECIDE, n_eigs, n_eigs);
      BVSetFromOptions(V_bv);
      
      BVMatMultHermitianTranspose(RR_bv, W_r_bv_Mat, V_bv);      
      
      BVRestoreMat(W_r_bv, &W_r_bv_Mat);  // restore the matrix so that we can reuse W_r_bv
      
      // Divide by -(L[n_eigs:].reshape([-1, 1]) - L[:n_eigs].conjugate())
      for(PetscInt column_idx = 0; column_idx < n_eigs; column_idx++)
      {
        Vec V_bv_col_Vec;
        BVGetColumn(V_bv, column_idx, &V_bv_col_Vec);
        
        for(PetscInt row_idx = 0; row_idx < n_eigs; row_idx++)
        {
          PetscScalar V_value;
          PetscScalar L_value_row_idx;  // L[row_idx + n_eigs]
          PetscScalar L_value_column_idx;  // L[column_idx]
          
          PetscInt row_idx_offset = row_idx + n_eigs; 
          
          VecGetValues(V_bv_col_Vec, 1, &row_idx, &V_value);
          VecGetValues(L_Vec, 1, &row_idx_offset, &L_value_row_idx);
          VecGetValues(L_Vec, 1, &column_idx, &L_value_column_idx);
          
          V_value = -V_value / (L_value_row_idx - std::conj(L_value_column_idx));
          
          VecSetValue(V_bv_col_Vec, row_idx, V_value, INSERT_VALUES);
        }
        
        BVRestoreColumn(V_bv, column_idx, &V_bv_col_Vec);
      }
      
      // Compute the new search space
      Mat V_bv_Mat;
      BVGetMat(V_bv, &V_bv_Mat);
      BVMult(R_bv, 1.0, 1.0, W_r_bv, V_bv_Mat);
      BVRestoreMat(V_bv, &V_bv_Mat); 
      
      // Restart T_bv
        
      // Update X part
      for(PetscInt column_idx = 0; column_idx < n_eigs; column_idx++)
      {
        Vec T_bv_column_Vec;
        BVGetColumn(T_bv, column_idx, &T_bv_column_Vec);  // get the column we want to replace in T_bv and link it to vector T_bv_column_Vec
        BVCopyVec(X_bv, column_idx, T_bv_column_Vec);     // copy the column of X_bv into the T_bv_column_Vec vector, effectively 
                                                          // copying the column of X_bv into the column of T_bv 
        BVRestoreColumn(T_bv, column_idx, &T_bv_column_Vec);
      }
      
      // Update S part
      for(PetscInt column_idx = 0; column_idx < n_eigs; column_idx++)
      {
        PetscInt column_idx_offset = column_idx + n_eigs;  // we need to offset because now we are updating the S part of T_bv
        
        Vec T_bv_column_Vec;
        BVGetColumn(T_bv, column_idx_offset, &T_bv_column_Vec);  // get the column we want to replace in T_bv and link it to vector T_bv_column_Vec
        BVCopyVec(R_bv, column_idx, T_bv_column_Vec);     // copy the column of R_bv into the T_bv_column_Vec vector, effectively 
                                                          // copying the column of R_bv into the column of T_bv 
                                                          // Note that the column index of R_bv is not offset, since R_bv 
                                                          // contains only the search space, only T_bv contains both the 
                                                          // (approximate) solution and the search space, hence the offset needed. 
        
        // If nothing is done, the search directions result in vectors with very
        // small norms. This leads to very badly conditioned reduced matrices. A
        // solution to this problem is to normalize these vectors associated to
        // the new search directions. This essentially means normalizing the columns
        // of S.
        if(normalize_S)
          VecNormalize(T_bv_column_Vec, NULL);
                                                            
        BVRestoreColumn(T_bv, column_idx_offset, &T_bv_column_Vec);
      } 
      
      // TODO Apply the projector to ensure X and S satisfy the
      // 
      // TODO Update T (or update during the code so that at the restart of the
      //      loop, we have the new T ready) (I think this is already done)

    } while((iter_idx <= 150) && (error_max > tol));
    
    
    // Mat R_Mat;  // get the matrix associated to the search space eigenvectors to use with mult below
    // BVGetMat(R_bv, &R_Mat); 
    // std::cout << "\n\nR_Mat:" << std::endl;
    // MatView(R_Mat, PETSC_VIEWER_STDOUT_WORLD);
    // BVRestoreMat(R_bv, &R_Mat);  
    
    std::cout << "\n**************************************************************" << std::endl;
    std::cout << "* Doehler eingenvalue solver (PETSc) END" << std::endl;
    std::cout << "**************************************************************\n" << std::endl;
  }
  
  void compute_residual_eigen_v(Mat &A_Mat, Mat &M_Mat, Vec &L_Vec, BV &X_bv, 
      PetscInt eigen_idx_start, PetscInt n_eigs, BV &R_bv, MPI_Comm comm)
  {
    //  Computes:
    //    R = A_Mat @ X_bv - M_Mat @ (X_bv * L)
    //   
    //  Where:
    //    L_vec: are the n eigenvalues of the generalised eigenvalue problem A X = L M X
    //    X_bv: are n vectors with the same dimensions as the eigenvectors of the generalised eigenvalue problem A X = L M X
    //    A_Mat: is the A matrix of the generalised eigenvalue problem
    //    M_Mat: is the M matrix of the generalised eigenvalue problem
    //    R_bv: are the n residual vectors, associated to each eigenvalue and eigenvector pair
    //    eigen_idx_start: the start index for the eigenvalues to use 
    //    n_eigs: the number of eigenvalues to use to compute the residual
    //            the eigenvalues will be in the range 
    //               [eigen_idx_start, eigen_idx_start + n_eigs] (excluding the last)
    //            NOTE: X_bv and R_bv must have n_eigs columns.
    
    // Compute X * L
    BV XL_bv;  // to hold X * L[:2*n_eigs]
    BVDuplicate(X_bv, &XL_bv);
    BVCopy(X_bv, XL_bv);  // start by copying X and then below we multiply each column by the eigen value
    
    for(PetscInt eigen_v_idx = eigen_idx_start; eigen_v_idx < (eigen_idx_start + n_eigs); eigen_v_idx++)
    {
      // The BV contains only the eigenvectors we want from [0, n_eigs]
      // but the eigenvalue vector L_Vec contains all the eigenvalues 
      // by providing the eigen_v_idx and n_eigs parameters we can select 
      // the eigenvalues of the vector L_vec. We then just need to match 
      // the indices as done below.
      PetscInt bv_column_idx = eigen_v_idx - eigen_idx_start;  // get the column index of the eigenvector 
                                                               // of the BV associated to eigenvalue eigen_v_idx
      PetscScalar eigen_value;
      VecGetValues(L_Vec, 1, &eigen_v_idx, &eigen_value);
      BVScaleColumn(XL_bv, bv_column_idx, eigen_value);
    }
    
    // Compute M @ (X * L)
    BV MXL_bv;  // to hold M @ (X * L)
    BVDuplicate(XL_bv, &MXL_bv);  
    BVMatMult(XL_bv, M_Mat, MXL_bv);  // make the multiplication M @ (X * L[eigen_idx_start:eigen_idx_end])
    
    // Compute A @ X
    // We use already R_bv, we then substract M @ (X * L) to get the final residual value
    BVMatMult(X_bv, A_Mat, R_bv);  // make the multiplication A @ X
    
    // Compute (A @ X) - (M @ (X * L))
    BVMult(R_bv, -1, 1.0, MXL_bv, NULL);
  }

  
  
  
}  // namespace doehler
