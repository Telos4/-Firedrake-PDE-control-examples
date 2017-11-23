from __future__ import absolute_import, division, print_function
from firedrake import *
from firedrake_adjoint import *
from os.path import abspath, basename, dirname, join
import numpy as np


class eomegaExpression(Expression):
    def eval(self, value, X):
        value[:] = 2.0*np.pi*np.pi*np.sin(np.pi*X[0])*np.sin(np.pi*X[1]) \
                   + np.sign(-np.sin(8.0*np.pi*X[0])*np.sin(8.0*np.pi*X[1]))

class ubarExpression(Expression):
    def eval(self, value, X):
        value[:] = - np.sign(-1/(128*np.pi*np.pi)*np.sin(8*np.pi*X[0])*np.sin(np.pi*8*X[1]))

class pde(object):
    # We solve the test problem
    # min_{u} \int_{\Omega} (y - y_{\Omega})^2 dx
    #  s.t.     - \Delta y = u + e_{\Omega}  on \Omega
    #                    y = 0               on \Gamma
    #               -1 <= u(x) <= 1
    # taken from [Tröltzsch, p.64] using a gradient projection method.
    def __init__(self, mesh):
        self.verbose = False
        self.mesh = mesh

        self.S = FunctionSpace(self.mesh, "CG", 1)
        self.ua = Function(self.S, name="ua")  # lower bound for control
        self.ua.assign(-1.0)
        self.ub = Function(self.S, name="ub")  # upper bound for control
        self.ub.assign(1.0)

        self.y_omega = Function(self.S, name="y_omega")     # reference solution
        self.y_omega.interpolate(Expression('sin(pi*x[0])*sin(pi*x[1])+sin(8*pi*x[0])*sin(8*pi*x[1])'))
        self.y_omega.rename("y_omega")

        self.e_omega = Function(self.S, name="e_omega")
        self.e_omega.rename("e_omega")
        self.e_omega.interpolate(eomegaExpression())

        self.pde_solver_parameters = {
            "mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_atol": 1e-10,
            "pc_type": "lu",
        }

        if self.verbose:
            self.pde_solver_parameters["snes_monitor"] = True
            self.pde_solver_parameters["ksp_converged_reason"] = True

        self.s = 0.5

    def setup_solver(self):

        # define functions
        v = TestFunction(self.S)

        self.y = Function(self.S, name="y")   # state variable y
        self.y.rename("y")

        self.u = Function(self.S, name="u")   # control variable u
        self.u.rename("u")



        # boundary conditions
        self.T_bcs = [DirichletBC(self.S, Constant(0.0), (1, 2, 3, 4))]


        # solver for pde
        self.a = inner(grad(self.y), grad(v)) * dx

        self.F = inner(self.u + self.e_omega, v) * dx

        self.pde_eq_problem = NonlinearVariationalProblem(self.a - self.F, self.y, self.T_bcs)

        self.pde_eq_solver = NonlinearVariationalSolver(
            self.pde_eq_problem,
            solver_parameters=self.pde_solver_parameters)


    def eval_J(self, y, u):
        return norm(y - self.y_omega)**2

    def optimize(self):

        #outfile = File(join(data_dir, "../", "results/", "ex1_all_iterates.pvd"))


        # 1. solve pde
        self.pde_eq_solver.solve()

        J = Functional(inner(self.y - self.y_omega, self.y - self.y_omega) * dx, name = "J")

        rf_J = ReducedFunctional(J, Control(self.u))

        #u_opt = Function(self.S, name = "u_opt")
        self.u.assign(minimize(rf_J, method = "L-BFGS-B", tol=1e-10, bounds = (self.ua, self.ub)))
        adj_reset()
        self.pde_eq_solver.solve()

        adj_html("forward.html", "forward")
        adj_html("adjoint.html", "adjoint")

        success = replay_dolfin(tol=0.0, stop=True)

        return self.y, self.u

if __name__ == "__main__":
    cwd = abspath(dirname(__file__))
    data_dir = join(cwd, "data")

    mesh = UnitSquareMesh(100,100)

    pde = pde(mesh)

    pde.setup_solver()

    outfile = File(join(data_dir, "../", "results/", "ex1_all_solution.pvd"))

    y, u = pde.optimize()

    # exact solutions
    ybar = Function(pde.S)
    ybar.interpolate(Expression('sin(pi*x[0])*sin(pi*x[1])'))
    ybar.rename('ybar')

    ubar = Function(pde.S)
    ubar.interpolate(ubarExpression())
    ubar.rename('ubar')

    # errors
    eu = Function(pde.S)
    eu.assign(assemble(u - ubar))
    eu.rename("error_u")
    ey = Function(pde.S)
    ey.assign(assemble(y - ybar))
    ey.rename("error_y")
    # ep = Function(pde.S)
    # ep.assign(p - pbar)
    # ep.rename("error_p")
    #
    # print("|y - y_bar| = " + str(norm(ey)))
    # print("|u - u_bar| = " + str(norm(eu)))
    # print("|p - p_bar| = " + str(norm(ep)))

    outfile.write(y, u, ybar, ubar, eu, ey)
