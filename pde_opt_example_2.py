from __future__ import absolute_import, division, print_function
from firedrake import *
from os.path import abspath, basename, dirname, join

class eomegaExpression(Expression):
    def eval(self, value, X):
        value[:] = 1.0 - min(1.0, max(0.0, 12.0 * ((X[0] - 0.5)**2 + (X[1] - 0.5)**2) - 1.0/3.0))

class ubarExpression(Expression):
    def eval(self, value, X):
        value[:] = min(1.0, max(0.0, 12.0 * ((X[0] - 0.5)**2 + (X[1] - 0.5)**2) - 1.0/3.0))

class pde(object):
    # We solve the test problem
    # min_{u} \frac{1}{2} \int_{\Omega} (y - y_{\Omega})^2 dx + \int_{\Gamma} e_{\Gamma} y ds
    #           + \frac{1}{2} \int_{\Omega} u^2 dx
    #  s.t.       - \Delta y + y = u + e_{\Omega}  on \Omega
    #           \partial_{\nu} y = 0               on \Gamma
    #                 0 <= u(x) <= 1
    # taken from [Tröltzsch, p.65] using a gradient projection method.
    def __init__(self, mesh):
        self.verbose = False
        self.mesh = mesh

        self.S = FunctionSpace(self.mesh, "CG", 1)

        self.ua = Function(self.S)  # lower bound for control
        self.ua.assign(0.0)
        self.ub = Function(self.S)  # upper bound for control
        self.ub.assign(1.0)

        self.y_omega = Function(self.S)     # reference solution
        self.y_omega.interpolate(Expression('-142.0/3.0 + 12.0 * ((x[0] - 0.5)*(x[0] - 0.5) + (x[1] - 0.5)*(x[1] - 0.5))'))
        self.y_omega.rename("y_omega")

        self.e_omega = Function(self.S)
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

        self.adj_solver_parameters = {
            "mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_atol": 1e-10,
            "pc_type": "lu",
        }

        if self.verbose:
            self.adj_solver_parameters["snes_monitor"] = True
            self.adj_solver_parameters["ksp_converged_reason"] = True

        self.s = 0.5

    def setup_solver(self):

        # define functions
        v = TestFunction(self.S)

        self.y = Function(self.S)   # state variable y
        self.y.rename("y")

        self.u = Function(self.S)   # control variable u
        self.u.rename("u")
        #self.u.interpolate(ubarExpression())

        self.p = Function(self.S)   # adjoint variable p
        self.p.rename("p")

        self.v = Function(self.S)   # descent direction
        self.v.rename("v")


        # solver for pde
        self.a = (inner(grad(self.y), grad(v)) + self.y * v) * dx

        self.F = inner(self.u + self.e_omega, v) * dx

        self.pde_eq_problem = NonlinearVariationalProblem(self.a - self.F, self.y)

        self.pde_eq_solver = NonlinearVariationalSolver(
            self.pde_eq_problem,
            solver_parameters=self.pde_solver_parameters)


        # solver for adjoint
        self.a_adj = (inner(grad(self.p), grad(v)) + self.p * v) * dx

        self.F_adj = inner(self.y - self.y_omega, v) * dx - Constant(12.0) * v * ds

        self.adj_eq_problem = NonlinearVariationalProblem(self.a_adj - self.F_adj, self.p)
        self.adj_eq_solver = NonlinearVariationalSolver(
            self.adj_eq_problem,
            solver_parameters=self.adj_solver_parameters
        )

    def eval_J(self, y, u):
        temp = assemble(self.e_omega * y * ds)
        return 0.5 * norm(y - self.y_omega)**2 + 0.5 * norm(u)**2 + temp

    def optimize(self):
        self.u_old = Function(self.S)

        outfile = File(join(data_dir, "../", "results/", "example_2_iterates.pvd"))

        k = 0
        k_max = 100

        # the algorithm uses a gradient projection method
        while True:
            # 1. solve pde
            self.pde_eq_solver.solve()

            f_old = self.eval_J(self.y, self.u)    # old value of the objective

            print("f(u) = J(y(u),u) = " + str(f_old))

            # 2. solve adjoint system
            self.adj_eq_solver.solve()

            outfile.write(self.y, self.p, self.u)

            # 3. compute descent direction
            self.v.assign(-(self.p + self.u))

            # 4. compute step length
            self.s = 0.5

            # 5. compute new iterate
            self.u_old = self.u.copy(deepcopy=True) # copy of old iterate

            project(Max(Min(self.u_old + self.s * self.v, self.ub), self.ua), self.u)

            delta_u = norm(self.u_old - self.u)
            grad_f = assemble((self.p + self.u) * (self.v - self.u) * dx)

            k += 1

            # check termination criterion
            if k > k_max:
                print("maximum number of iterations reached. terminating.")
                break
            elif delta_u < 1.0e-5:
                print("no change in iterates anymore. terminating.")
                break
            elif grad_f >= 0.0:
                print("variational equation satisfied. terminating.")
                break

        print("iterations: " + str(k))
        print("|u^{k+1} - u^k| = " + str(delta_u))
        print("f'(u_n)(v_n - u_n) = " + str(grad_f))

        return self.y, self.u, self.p

if __name__ == "__main__":
    cwd = abspath(dirname(__file__))
    data_dir = join(cwd, "data")

    mesh = UnitSquareMesh(100,100)

    pde = pde(mesh)

    pde.setup_solver()

    outfile = File(join(data_dir, "../", "results/", "example_2_solution.pvd"))

    y, u, p = pde.optimize()

    # exact solutions
    ybar = Function(pde.S)
    ybar.interpolate(Expression('1.0'))
    ybar.rename('ybar')

    pbar = Function(pde.S)
    pbar.interpolate(Expression('-12.0 * ((x[0] - 0.5)*(x[0] - 0.5) + (x[1] - 0.5)*(x[1] - 0.5)) + 1.0/3.0'))
    pbar.rename('pbar')

    ubar = Function(pde.S)
    ubar.interpolate(ubarExpression())
    ubar.rename('ubar')

    # errors
    eu = Function(pde.S)
    eu.assign(u - ubar)
    eu.rename("error_u")
    ey = Function(pde.S)
    ey.assign(y - ybar)
    ey.rename("error_y")
    ep = Function(pde.S)
    ep.assign(p - pbar)
    ep.rename("error_p")

    print("|y - y_bar| = " + str(norm(ey)))
    print("|u - u_bar| = " + str(norm(eu)))
    print("|p - p_bar| = " + str(norm(ep)))

    outfile.write(y,p,u, ybar, pbar, ubar, eu, ey, ep)