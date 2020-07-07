#include <Eigen/StdVector>
#include <Eigen/Core>

#include <iostream>
#include <stdint.h>

#include <unordered_set>
#include <memory>
#include <vector>
#include <stdlib.h> 

#include "g2o/stuff/sampler.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#include "g2o/solvers/structure_only/structure_only_solver.h"

#include "common/BundleParams.h"
#include "common/BALProblem.h"
#include "g2o_bal_class.h"

using namespace Eigen;
using namespace std;

typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;
typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BalBlockSolver;

// set up the vertexs and edges for the bundle adjustment. 
void BuildProblem( const BALProblem* bal_problem, 
   g2o::SparseOptimizer* optimizer,
   const BundleParams& params)
{
  const int num_points = bal_problem->num_points();
  const int num_cameras = bal_problem->num_cameras();
  const int camera_block_size = bal_problem->camera_block_size();
  const int point_block_size = bal_problem->point_block_size();
  
  // set camera vertetx with initial value in the dataset.
  const double* raw_cameras = bal_problem->cameras();
  for(int i = 0; i < num_cameras; ++i) 
  {
    ConstVectorRef temVecCamera();
    
  }
  
  
  
  
}































