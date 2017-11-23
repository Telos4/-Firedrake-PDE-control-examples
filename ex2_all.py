from __future__ import absolute_import, division, print_function
from firedrake import *
from firedrake_adjoint import *
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

        self.ua = Function(self.S, name="ua")  # lower bound for control
        self.ua.assign(0.0)
        self.ub = Function(self.S, name="ub")  # upper bound for control
        self.ub.assign(1.0)

        self.y_omega = Function(self.S, name="y_omega")     # reference solution
        self.y_omega.interpolate(Expression('-142.0/3.0 + 12.0 * ((x[0] - 0.5)*(x[0] - 0.5) + (x[1] - 0.5)*(x[1] - 0.5))'))
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
        #self.u.assign(Constant(1))
        self.u.interpolate(ubarExpression())

        # solver for pde
        self.a = (inner(grad(self.y), grad(v)) + self.y * v) * dx

        self.F = inner(self.u + self.e_omega, v) * dx

        self.pde_eq_problem = NonlinearVariationalProblem(self.a - self.F, self.y)

        self.pde_eq_solver = NonlinearVariationalSolver(
            self.pde_eq_problem,
            solver_parameters=self.pde_solver_parameters)


    def eval_J(self, y, u):
        temp = assemble(self.e_omega * y * ds)
        return 0.5 * norm(y - self.y_omega)**2 + 0.5 * norm(u)**2 + temp

    def optimize(self):

        self.pde_eq_solver.solve()
        #success = replay_dolfin(tol=0.0, stop=True)
        #print("model right %s" %success)


        #temp = assemble(self.e_omega * self.y * ds)
        J = Functional(0.5*inner(self.y - self.y_omega, self.y - self.y_omega) * dx + 0.5*inner(self.u, self.u) * dx + (self.e_omega * self.y) * ds , name = "J")


        controls = File(join(data_dir, "../", "results/", "ex2_all_iterates_optimisation.pvd"))
        a_viz = Function(self.S, name="ControlVisualisation")
        j_viz = Function(self.S, name="jVisualisation")
        dj_viz = Function(self.S, name="djVisualisation")

        def derivative_cb(j):
            #a_viz.assign(a)
            j_viz.assign(j)
            #dj_viz.assign(dj)

            #controls.write(a_viz, j_viz, dj_viz)
            controls.write(j_viz)



        rf_J = ReducedFunctional(J, Control(self.u), derivative_cb_pre = derivative_cb)
        self.u.assign(minimize(rf_J, method = "TNC", tol=1e-5, bounds = (self.ua, self.ub)))

        self.pde_eq_solver.solve()

        adj_html("forward.html", "forward")
        adj_html("adjoint.html", "adjoint")

        #success = replay_dolfin(tol=0.0, stop=False)

        #print(self.eval_J(self.y, self.u))
        return self.y, self.u

if __name__ == "__main__":
    cwd = abspath(dirname(__file__))
    data_dir = join(cwd, "data")

    mesh = UnitSquareMesh(100,100)

    pde = pde(mesh)

    pde.setup_solver()

    outfile = File(join(data_dir, "../", "results/", "ex2_all_solution.pvd"))

    y, u = pde.optimize()

    # exact solutions
    ybar = Function(pde.S, name="ybar")
    ybar.interpolate(Expression('1.0'))
    ybar.rename('ybar')

    ubar = Function(pde.S, name="ubar")
    ubar.interpolate(ubarExpression())
    ubar.rename('ubar')

    # errors
    eu = Function(pde.S, name="eu")
    eu.assign(assemble(u - ubar))
    eu.rename("error_u")
    ey = Function(pde.S, name="ey")
    ey.assign(assemble(y - ybar))
    ey.rename("error_y")

    outfile.write(y, u, ybar, ubar, eu, ey)
