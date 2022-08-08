// Copyright (C) 2016-2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGS_H
#define LBFGS_H

#include "Eigen/Core"
#include "LBFGSpp/Param.h"
#include "LBFGSpp/BFGSMat.h"
#include "LBFGSpp/LineSearchBacktracking.h"
#include "LBFGSpp/LineSearchBracketing.h"
#include "LBFGSpp/LineSearchNocedalWright.h"

//boost bimap
#include <boost/bimap.hpp>
#include <boost/bimap/unconstrained_set_of.hpp>
#include <boost/bimap/multiset_of.hpp>
#include <boost/bimap/support/lambda.hpp>
typedef boost::bimap<std::string,double> bimap;


//autodiff
#include <boost/math/differentiation/autodiff.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>


namespace LBFGSpp {


///
/// L-BFGS solver for unconstrained numerical optimization
///
template < typename Scalar,
           template<class> class LineSearch = LineSearchBacktracking >
class LBFGSSolver
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<Vector> MapVec;

    const LBFGSParam<Scalar>& m_param;  // Parameters to control the LBFGS algorithm
    BFGSMat<Scalar>           m_bfgs;   // Approximation to the Hessian matrix
    Vector                    m_fx;     // History of the objective function values
    Vector                    m_xp;     // Old x
    Vector                    m_grad;   // New gradient
    Vector                    m_gradp;  // Old gradient
    Vector                    m_drt;    // Moving direction
    Vector                    prev_x;    // Moving direction
    double                    f_prev;    // Moving direction
    double                    prev_gnorm = 9999;    // Moving direction

    // Reset internal variables
    // n: dimension of the vector to be optimized
    inline void reset(int n)
    {
        const int m = m_param.m;
        m_bfgs.reset(n, m);
        m_xp.resize(n);
        m_grad.resize(n);
        m_gradp.resize(n);
        m_drt.resize(n);
        if(m_param.past > 0)
            m_fx.resize(m_param.past);
        f_prev = 0;
        prev_x.resize(n);
    }

public:
    ///
    /// Constructor for the L-BFGS solver.
    ///
    /// \param param An object of \ref LBFGSParam to store paremtrs for the
    ///        algorithm
    ///
    LBFGSSolver(const LBFGSParam<Scalar>& param) :
        m_param(param)
    {
        m_param.check_param();
    }

    ///
    /// Minimizing a multivariate function using the L-BFGS algorithm.
    /// Exceptions will be thrown if error occurs.
    ///
    /// \param f  A function object such that `f(x, grad)` returns the
    ///           objective function value at `x`, and overwrites `grad` with
    ///           the gradient.
    /// \param x  In: An initial guess of the optimal point. Out: The best point
    ///           found.
    /// \param fx Out: The objective function value at `x`.
    ///
    /// \return Number of iterations used.
    ///
    template <typename Foo>
    inline int minimize(Foo& f, Vector& x, Scalar& fx, ptr_bowl &parameters,col_vector &xstar)
    {
        using std::abs;

        // Dimension of the vector
        const int n = x.size();
        reset(n);

        // The length of lag for objective function value to test convergence
        const int fpast = m_param.past;

        // Evaluate function and compute gradient
        fx = f(x, m_grad,parameters,xstar);
        Scalar gnorm = m_grad.norm();
        if(fpast > 0)
            m_fx[0] = fx;

        // Early exit if the initial x is already a minimizer
        if(gnorm <= m_param.epsilon || gnorm <= m_param.epsilon_rel * x.norm())
        {
            return 1;
        }

        // Initial direction
        m_drt.noalias() = -m_grad;
        // Initial step size
        Scalar step = Scalar(1) / m_drt.norm();

        // Number of iterations used
        int k = 1;
        for( ; ; )
        {
            // Save the curent x and gradient
            m_xp.noalias() = x;
            m_gradp.noalias() = m_grad;

            // Line search to update x, fx and gradient
            f_prev = fx;
            prev_x = x;
            LineSearch<Scalar>::LineSearch(f, fx, x, m_grad, step, m_drt, m_xp, m_param,parameters,xstar);
            (*parameters).optim->n_iter++; 

            // New gradient norm
            gnorm = m_grad.norm();
            

            // Convergence test -- gradient

            /*
            std::cout << "Test" << gnorm << std::endl;
            std::cout << m_param.epsilon << std::endl;
            std::cout <<  x.norm() << std::endl;
            std::cout << m_param.epsilon_rel * x.norm() << std::endl;
            */

            //is added: gnorm < 0.2 before

            if(gnorm <= m_param.epsilon || gnorm <= m_param.epsilon_rel * x.norm())
            {
                std::cout << "Criteria is satisfied with a good optimum"; 
                return k;
            }

            // Maximum number of iterations
            if(m_param.max_iterations != 0 && k >= m_param.max_iterations)
            {
                std::cout << "Criteria is satisfied with a good optimum"; 
                return k;
            }

            if(fpast > 0)
            {
                const Scalar fxd = m_fx[k % fpast];
                if(k >= fpast && abs(fxd - fx) <= m_param.delta * std::max(std::max(abs(fx), abs(fxd)), Scalar(1))){
                    std::cout << "Criteria is satisfied with a good optimum"; 
                    return k;
                }

                m_fx[k % fpast] = fx;
            }

            if(true){
                int s1 = (*parameters).y_size, s2 = (*parameters).y_size, s3 = (*parameters).num_Con;
                double threshold1 = (1e-3)*sqrt(s1),  threshold2 = (1e-3)*sqrt(s1), threshold3 = 5e-3*sqrt(s1), threshold4 =  4e-3*sqrt(s1); //0.03*sqrt((*parameters).num_Con);

                std::cout << "fx = " << fx <<  " ,|gnorm| " << abs(gnorm) << ", |x - x.old|: " << abs((x-prev_x).norm()) << ", |f - f.old|: " << abs(fx -f_prev) << std::endl;
                //std::cout << std::endl;

                if(abs(gnorm) < threshold1 && abs((x-prev_x).norm())< threshold2) {std::cout << "Criteria is satisfied with a good optimum"; return k;}
                if(abs(gnorm) < threshold1 && abs(fx -f_prev)< threshold1) {std::cout << "Criteria is satisfied with a good optimum"; return k;}
                if(abs((x-prev_x).norm())< threshold2 && abs(fx -f_prev)< threshold1) {std::cout << "Criteria is satisfied with a good optimum"; return k;}
                if(abs(gnorm) < 1e-1 && abs((x-prev_x).norm())< 1e-2 && abs(fx -f_prev)<1e-3) {std::cout << "Criteria is satisfied with a good optimum"; return k;}
                //if(abs(gnorm) < threshold4 && abs((x-prev_x).norm())< threshold4 && abs(fx -f_prev)<threshold4) {std::cout << "Criteria 4 is satisfied" << std::endl; return k;}
                //if(abs(gnorm) < 1.8 && abs((x-prev_x).norm())< 0.6 && abs(fx -f_prev)<0.6) {std::cout << "Criteria 4 is satisfied" << std::endl; return k;}
           
            }else{

                double stop_threshold = 1e-3*std::sqrt(m_param.data_size); //previously datasize is xsize + ysize and it was 1e-4
                std::cout << "threshold: " << stop_threshold <<  ", |gnorm| " << abs(gnorm) << ", |x - x.old|: " << abs((x-prev_x).norm()) << ", |f - f.old|: " << abs(fx -f_prev) << std::endl;
                std::cout << std::endl;
                //if(abs(gnorm) < stop_threshold || abs((x-prev_x).norm())< stop_threshold || abs(fx -f_prev)< stop_threshold) return k;
                
                if(abs(gnorm) < stop_threshold && abs((x-prev_x).norm())< stop_threshold) return k;
                if(abs(gnorm) < stop_threshold && abs(fx -f_prev)< stop_threshold) return k;
                if(abs((x-prev_x).norm())< stop_threshold && abs(fx -f_prev)< stop_threshold) return k;
                if(abs(gnorm) < 1e-1 && abs((x-prev_x).norm())< 1e-2 && abs(fx -f_prev)<1e-3) return k;
            }
            
            

           

            //if(abs(gnorm) < 0.5 && prev_gnorm < 0.5 && abs((x-prev_x).norm())< 1e-2 && abs(fx -f_prev)<1e-3) return k;
            //prev_gnorm = abs(gnorm);

            // Update s and y
            // s_{k+1} = x_{k+1} - x_k
            // y_{k+1} = g_{k+1} - g_k
            m_bfgs.add_correction(x - m_xp, m_grad - m_gradp);

            // Recursive formula to compute d = -H * g
            m_bfgs.apply_Hv(m_grad, -Scalar(1), m_drt);

            // Reset step = 1.0 as initial guess for the next line search
            step = Scalar(1);
            k++;
        }

        return k;
    }
};


} // namespace LBFGSpp

#endif // LBFGS_H
