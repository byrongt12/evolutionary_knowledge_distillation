#
#  * This class implements a problem domain and implements ProblemDomain abstract
#  * class.
#  *
#  * N. Pillay
#  *
#  * 30 August 2016 
#  

from GeneticAlgorithm.Problem import Problem
from GeneticAlgorithm.distillation.DistillationSolution import DistillationSolution


class DistillationProblem(Problem):
    # Methods that are abstract in ProblemDomain that need to be implemented
    def evaluate(self, heuristic_combination, trainingItems):
        # Implements the abstract method to create a solution using heuristicComb
        # using an instance of the Solution class which is also used to calculate
        # the fitness using the objective value of the created solution.
        solution = DistillationSolution()
        solution.set_heuristic_combination(heuristic_combination)
        solution.create_solution(trainingItems)
        return solution
