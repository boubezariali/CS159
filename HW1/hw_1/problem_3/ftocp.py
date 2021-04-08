import pdb
import numpy as np
from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from scipy import linalg
from scipy import sparse
from cvxopt.solvers import qp
import datetime
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP
from dataclasses import dataclass, field


class FTOCP(object):
	""" Finite Time Optimal Control Problem (FTOCP)
	Methods:
		- solve: solves the FTOCP given the initial condition x0 and terminal contraints
		- buildNonlinearProgram: builds the ftocp program solved by the above solve method
		- model: given x_t and u_t computes x_{t+1} = f( x_t, u_t )

	"""

	def __init__(self, N, A, B, C, Q, R, Qf, Fx, bx, Fu, bu, Ff, bf, printLevel):
		# Define variables
		self.printLevel = printLevel

		self.A  = A
		self.B  = B
		self.C  = C
		self.N  = N
		self.n  = A[0].shape[1]
		self.d  = B[0].shape[1]
		self.Fx = Fx
		self.bx = bx
		self.Fu = Fu
		self.bu = bu
		self.Ff = Ff
		self.bf = bf
		self.Q  = Q
		self.Qf = Qf
		self.R  = R

		print("Initializing FTOCP")
		self.buildIneqConstr()
		self.buildCost()
		self.buildEqConstr()
		print("Done initializing FTOCP")

		self.time = 0


	def solve(self, x0):
		"""Computes control action
		Arguments:
		    x0: current state
		"""

		# Solve QP
		startTimer = datetime.datetime.now()
		self.osqp_solve_qp(self.H, self.q, self.G_in, np.add(self.w_in, np.dot(self.E_in,x0)), self.G_eq, np.add(np.dot(self.E_eq,x0),self.C_eq))
		endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
		self.solverTime = deltaTimer

		# Unpack Solution
		self.unpackSolution(x0)

		self.time += 1

		return self.uPred[0,:]

	def unpackSolution(self, x0):
		# Extract predicted state and predicted input trajectories
		self.xPred = np.vstack((x0, np.reshape((self.Solution[np.arange(self.n*(self.N))]),(self.N,self.n))))
		self.uPred = np.reshape((self.Solution[self.n*(self.N)+np.arange(self.d*self.N)]),(self.N, self.d))

		if self.printLevel >= 2:
			print("Optimal State Trajectory: ")
			print(self.xPred)

			print("Optimal Input Trajectory: ")
			print(self.uPred)

		if self.printLevel >= 1: print("Solver Time: ", self.solverTime.total_seconds(), " seconds.")

	def buildIneqConstr(self):
		# Hint: Are the matrices G_in, E_in and w_in constructed using A and B?
		G_in = linalg.block_diag(*([self.Fx]*(self.N-1) + [self.Ff] + [self.Fu]*self.N))
		G_in = np.vstack([np.zeros((self.Fx.shape[0], G_in.shape[1])), G_in])

		nbx = self.bx.shape[0]
		nbf = self.bf.shape[0]
		nbu = self.bu.shape[0]
		zeros_dim = (nbx * (self.N) + nbf + nbu * self.N) - self.Fx.T.shape[1]
		E_in = np.hstack([-self.Fx.T, np.zeros((self.Fx.T.shape[0], zeros_dim))])
		E_in = E_in.T

		w_in = np.hstack([self.bx.T]*(self.N) + [self.bf.T] + [self.bu.T]*self.N).T

		if self.printLevel >= 2:
			print("G_in: ")
			print(G_in)
			print("E_in: ")
			print(E_in)
			print("w_in: ", w_in)

		self.G_in = sparse.csc_matrix(G_in)
		self.E_in = E_in
		self.w_in = w_in.T

	def buildCost(self):
		# Hint: Are the matrices H and q constructed using A and B?
		barQ = linalg.block_diag(*([self.Q] * (self.N-1) + [self.Qf]))
		barR = linalg.block_diag(*([self.R] * self.N))

		H = linalg.block_diag(barQ, barR)
		q = np.zeros(H.shape[0])

		if self.printLevel >= 2:
			print("H: ")
			print(H)
			print("q: ", q)

		self.q = q
		self.H = sparse.csc_matrix(2 * H)  #  Need to multiply by two because CVX considers 1/2 in front of quadratic cost

	def buildEqConstr(self):
		Id = linalg.block_diag(*([np.identity(self.A[0].shape[0])] * self.N))
		A_block = linalg.block_diag(*self.A)
		A_block = np.roll(A_block, -self.A[0].shape[0], axis=1)  # Shift downwards
		A_block[:self.A[0].shape[0],:] = 0  # Clear rolled over at the top
		left = Id - A_block
		right = linalg.block_diag(*self.B)
		assert(left.shape[0] == right.shape[0]), (left.shape, right.shape)
		G_eq = np.hstack([left, -right])

		E_eq = np.hstack([self.A[0].T] + [np.zeros(self.A[0].T.shape)] * (self.N-1)).T

		C_eq = np.hstack(self.C)

		if self.printLevel >= 2:
			print("G_eq: ")
			print(G_eq)
			print("E_eq: ")
			print(E_eq)
			print("C_eq: ", C_eq)

		self.C_eq = C_eq
		self.G_eq = sparse.csc_matrix(G_eq)
		self.E_eq = E_eq

	def osqp_solve_qp(self, P, q, G= None, h=None, A=None, b=None, initvals=None):
		"""
		Solve a Quadratic Program defined as:
		minimize
			(1/2) * x.T * P * x + q.T * x
		subject to
			G * x <= h
			A * x == b
		using OSQP <https://github.com/oxfordcontrol/osqp>.
		"""

		qp_A = vstack([G, A]).tocsc()
		l = -inf * ones(len(h))
		qp_l = hstack([l, b])
		qp_u = hstack([h, b])

		self.osqp = OSQP()
		self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)

		if initvals is not None:
			self.osqp.warm_start(x=initvals)
		res = self.osqp.solve()
		if res.info.status_val == 1:
			self.feasible = 1
		else:
			self.feasible = 0
			print("The FTOCP is not feasible at time t = ", self.time)

		self.Solution = res.x
