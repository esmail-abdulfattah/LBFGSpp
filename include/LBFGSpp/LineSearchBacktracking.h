// Copyright (C) 2016-2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LINE_SEARCH_BACKTRACKING_H
#define LINE_SEARCH_BACKTRACKING_H

#include "Eigen/Core"
#include <stdexcept>  // std::runtime_error


namespace LBFGSpp {


///
/// The backtracking line search algorithm for L-BFGS. Mainly for internal use.
///
template <typename Scalar>
class LineSearchBacktracking
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

public:
    ///
    /// Line search by backtracking.
    ///
    /// \param f      A function object such that `f(x, grad)` returns the
    ///               objective function value at `x`, and overwrites `grad` with
    ///               the gradient.
    /// \param fx     In: The objective function value at the current point.
    ///               Out: The function value at the new point.
    /// \param x      Out: The new point moved to.
    /// \param grad   In: The current gradient vector. Out: The gradient at the
    ///               new point.
    /// \param step   In: The initial step length. Out: The calculated step length.
    /// \param drt    The current moving direction.
    /// \param xp     The current point.
    /// \param param  Parameters for the LBFGS algorithm
    ///
    template <typename Foo>
    static void LineSearch(Foo& f, Scalar& fx, Vector& x, Vector& grad,
                           Scalar& step,
                           const Vector& drt, const Vector& xp,
                           const LBFGSParam<Scalar>& param, ptr_bowl &parameters, col_vector &xstar)
    { 
        // Decreasing and increasing factors
        const Scalar dec = 0.9;//0.5;
        const Scalar inc = 2.1;//2.1;

        // Check the value of step
        if(step <= Scalar(0))
            std::invalid_argument("'step' must be positive");

        // Save the function value at the current x
        const Scalar fx_init = fx;
        // Projection of gradient on the search direction
        const Scalar dg_init = grad.dot(drt);
        // Make sure d points to a descent direction
        if(dg_init > 0)
            std::logic_error("the moving direction increases the objective function value");

        const Scalar test_decr = param.ftol * dg_init;
        Scalar width;

        int iter;
        int m = 0;
        
        /*
        if(k>10) m = 2;
        else if(k>10) m = 2;
        else if(k>20) m = 5;
        else if(k>30) m = 20;*/

        for(iter = 0; iter < (param.max_linesearch + m); iter++)
        {

            // x_{k+1} = x_k + step * d_k
            x.noalias() = xp + step * drt;
            // Evaluate this candidate
            //std::cout << x << std::endl;
            fx = f(x, grad,parameters,xstar);

            if(fx > fx_init + step * test_decr)
            {
                width = dec;
            } else {
                // Armijo condition is met
                if(param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
                    break;

                const Scalar dg = grad.dot(drt);
                if(dg < param.wolfe * dg_init)
                {
                    width = inc;
                } else {
                    // Regular Wolfe condition is met
                    if(param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE)
                        break;

                    if(dg > -param.wolfe * dg_init)
                    {
                        width = dec;
                    } else {
                        // Strong Wolfe condition is met
                        break;
                    }
                }
            }

            
            if(iter >= param.max_linesearch)
                throw std::runtime_error("the line search routine reached the maximum number of iterations");

            if(step < param.min_step)
                throw std::runtime_error("the line search step became smaller than the minimum value allowed");

            if(step > param.max_step)
                throw std::runtime_error("the line search step became larger than the maximum value allowed");

            step *= 1.1*width;
        }
    }
};


} // namespace LBFGSpp

#endif // LINE_SEARCH_BACKTRACKING_H
