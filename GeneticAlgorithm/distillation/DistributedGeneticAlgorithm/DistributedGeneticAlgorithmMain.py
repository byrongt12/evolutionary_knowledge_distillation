#
#  * This is an example program used to illustrate how to use the EvoHyp library
#  * to create an initial solution to a problem using the genetic algorithm
#  * hyper-heuristic provided by the library.
#  *
#  * N. Pillay
#  *
#  * 30 August 2016
#
import time

from GeneticAlgorithm.DistributedGeneticAlgorithm import DistributedGeneticAlgorithm
from GeneticAlgorithm.distillation.DistillationProblem import DistillationProblem
from GeneticAlgorithm.distillation.DistillationSolution import DistillationSolution


class DistributedGeneticAlgorithmMain(object):
    @classmethod
    def solve(cls):
        # This method illustrates how the selection construction hyper-heuristic in
        # the GeneticAlgorithm library can be used to solve a combinatorial optimization problem.
        problem = DistillationProblem()
        seed = round(time.time() * 1000)
        heuristics = str("abc")
        distributed_genetic_algorithm = DistributedGeneticAlgorithm(seed, heuristics, 4)
        distributed_genetic_algorithm.set_parameters("Parameters.txt")
        distributed_genetic_algorithm.set_problem(problem)
        solution = distributed_genetic_algorithm.evolve()
        print("Best Solution")
        print("--------------")
        print("Fitness: ", solution.get_fitness())
        print("Heuristic combination: ", solution.get_heuristic_combination())
        print("Solution: ")
        DistributedGeneticAlgorithmMain.display_solution(solution)

    @classmethod
    def display_solution(cls, solution: DistillationSolution):
        # Displays the solution.
        print(' '.join(solution.get_solution()))

    @classmethod
    def main(cls):
        cls.solve()


if __name__ == '__main__':
    DistributedGeneticAlgorithmMain.main()
