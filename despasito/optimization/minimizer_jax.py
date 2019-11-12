"""
Authors: Andrew Abi-Mansour <andrew.gaam [at] gmail [dot] com>
         Pierre Ablin <pierreablin [at] gmail [dt] com>

License: MIT

"""
import autograd.numpy as np
import autograd
from scipy import optimize
import test_optim

class Minimizer:
    """A wrapper class for scipy.optimize.minimize that computes derivatives with JAX (AD).
        Parameters
        ----------
        objective_function : callable
            The objective function to be minimized.
                ``fun(optim_vars, *args) -> float``
            or
                ``fun(*optim_vars, *args) -> float``
            where optim_vars is either a numpy array or a list of numpy
            arrays and `args` is a tuple of fixed parameters needed to
            completely specify the function.
        optim_vars : ndarray or list of ndarrays
            Initial guess.
        args : tuple, optional
            Extra arguments passed to the objective function.
        precon_fwd : callable, optional
            The forward preconditioning.
                ``fun(optim_vars, *args) -> precon_optim_vars``
            or
                ``fun(*optim_vars, *args) -> precon_optim_vars``
            where optim_vars is either a numpy array or a list of numpy
            arrays and `args` is a tuple of fixed parameters needed to
            completely specify the function.
            The optimized function will be the composition:
            `objective_function(precon_fwd(optim_vars))`.
        precon_bwd : callable, optional
            The backward preconditioning.
                ``fun(precon_optim_vars, *args) -> optim_vars``
            or
                ``fun(*precon_optim_vars, *args) -> optim_vars``
            where optim_vars is either a numpy array or a list of numpy
            arrays and `args` is a tuple of fixed parameters needed to
            completely specify the function.
            This should be the reciprocal function of precon_fwd.
        kwargs : dict, optional
            Extra arguments passed to scipy.optimize.minimize. See
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            for the full list of available keywords.
        Returns
        -------
        output : ndarray or list of ndarrays
            The solution, of same shape as the input `optim_vars`.
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.
        """

    def __init__(self, objective_function, optim_vars, args=(), precon_fwd=None,
             precon_bwd=None, **kwargs):

        # Check if there is preconditioning:
        self._precondition = precon_fwd is not None
        if self._precondition != (precon_bwd is not None):

            error_string = {True: 'precon_fwd', False: 'precon_bwd'}[self._precondition]
            raise ValueError(f'You should specify both precon_fwd and precon_bwd,'
                             ' you only specified {error_string}')

        self._optim_vars = optim_vars
        self._objective_function = objective_function
        self._args = args
        self._precon_fwd = precon_fwd
        self._precon_bwd = precon_bwd
        self._kwargs = kwargs

    @staticmethod
    def _vectorize(optim_vars):
        shapes = [var.shape for var in optim_vars]
        x = np.concatenate([var.ravel() for var in optim_vars])
        return x, shapes

    @staticmethod
    def _split(x, shapes):
        x_split = np.split(x, np.cumsum([np.prod(shape) for shape in shapes[:-1]]))
        optim_vars = [var.reshape(*shape) for (var, shape) in zip(x_split, shapes)]
        return optim_vars

    def _objFunc(self, x):
        def _scipy_func(objective_function, gradient, x, shapes, args=()):
            """ whatever ... """
            optim_vars = Minimizer._split(x, shapes)
            obj = objective_function(optim_vars, *args)
            gradients = gradient(optim_vars, *args)
            g_vectorized, _ = Minimizer._vectorize(gradients)
        
            return obj, g_vectorized

        return _scipy_func(self.objective_converted, self._gradient, x, self._shapes, self._args)

    def minimize(self, **kwargs):

        if 'objective_function' in kwargs:
            self._objective_function = kwargs['objective_function']

        if 'optim_vars' in kwargs:
            self._optim_vars = kwargs['optim_vars']

        if 'args' in kwargs:
            self._args = kwargs['args']

        if self._precondition:

            self._optim_vars = self._convert_to_tuple(self._optim_vars)
            self._args = args
            precon_optim_vars = precon_fwd(*self._optim_vars, *args)

            precon_result, res = self.minimize(
                objective_function=self.precon_objective, 
                optim_vars = precon_optim_vars,
                args=args, precon_fwd=None, precon_bwd=None, 
                **kwargs)

            precon_result = self._convert_to_tuple(precon_result)
            return precon_bwd(*precon_result, *args), res

        # Check if there are bounds:
        bounds = self._kwargs.get('bounds')
        bounds_in_kwargs = bounds is not None

        # Convert input to a list if it is a single array
        if type(self._optim_vars) is np.ndarray:
            input_is_array = True
            self._optim_vars = (self._optim_vars,)
            if bounds_in_kwargs:
                bounds = (bounds,)
        else:
            input_is_array = False

        # Compute the gradient
        self._gradient = autograd.grad(self.objective_converted)
        # Vectorize optimization variables
        x0, self._shapes = Minimizer._vectorize(self._optim_vars)

        # Convert bounds to the correct format
        if bounds_in_kwargs:
            bounds = self._convert_bounds(bounds, shapes)
            self._kwargs['bounds'] = bounds

        print('x0 = ', x0)

        res = optimize.minimize(self._objFunc, x0, jac=True, **self._kwargs)

        # Convert output to the input format
        output = Minimizer._split(res['x'], self._shapes)
        if input_is_array:
            output = output[0]

        return output, res

    def precon_objective(self, *precon_optim_vars_and_args):
        """ whatever ... """

        args = precon_optim_vars_and_args[-len(self._args):]
        optim_vars = precon_bwd(*precon_optim_vars_and_args)
        optim_vars = self._convert_to_tuple(optim_vars)
            
        return objective_function(*optim_vars, *args)

    def _convert_to_tuple(self, optim_vars):
        if type(optim_vars) not in (list, tuple):
            return (optim_vars,)
    
        return optim_vars

    def objective_converted(self, optim_vars, *args):
        """ Converts loss to readable autograd format """
        return self._objective_function(*optim_vars, *args)

    def _convert_bounds(self, bounds, shapes):
        output_bounds = []
        for shape, bound in zip(shapes, bounds):
            # Check is the bound is already parsable by scipy.optimize
            b = bound[0]
            if isinstance(b, (list, tuple, np.ndarray)):
                output_bounds += bound
            else:
                output_bounds += [bound, ] * np.prod(shape)
        return output_bounds


# Performance test function
def objFunc(x):
    return optimize.rosen(x)



if __name__ == '__main__':
    
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
 
    Min = Minimizer(objFunc, x0, options={'disp': False})
    x_min, _ = Min.minimize()
    
    x_min_scipy = optimize.minimize(objFunc, x0, method='nelder-mead',
                options={'disp': False})

    print('output:', x_min, x_min_scipy.x)
