#
#  * This class implements the Solution abstract class and is used to store
#  * details of the initial solution.
#  *
#  * Nelishia Pillay
#  *
#  * 30 August 2016
#
from typing import List

import torch
from os import path

from GeneticAlgorithm.Solution import Solution
from NeuralNetwork.Train import train_model_with_distillation_only, evaluate


class DistillationSolution(Solution):
    # Data elements
    # Stores the heuristic combination that will be used to create an initial
    # solution.    
    heuristic_combination: str = ''

    # Stores the fitness value to be used for the initial solution created.
    fitness: float

    # Stores the initial solution created using the heuristic. In this problem
    # this is stored as an array of strings just as an example. However, the 
    # solution can be of any type, e.g. for the travelling salesman problem it 
    # could be a string representing the tour.
    initial_solution: List[str] = []

    # It may be necessary to store other values that are specific to problem being
    # solved that is different from the fitness or needed to calculate the fitness.
    # For example, for the examination timetabling problem the hard and soft
    # constraint cost also needs to be stored.
    # Implementation of abstract methods needed to extend Solution
    def get_fitness(self) -> float:
        return self.fitness

    def set_heuristic_combination(self, heuristic_combination: str):
        # Implements the abstract method to store the heuristic combination used to
        # create an initial solution.
        self.heuristic_combination = heuristic_combination

    def get_heuristic_combination(self) -> str:
        # Implements the abstract method to return the heuristic combination used to
        # create the solution.
        return self.heuristic_combination

    def get_solution(self) -> List[str]:
        # Implements the abstract method to return a solution.
        return self.initial_solution

    def fitter(self, other: Solution):
        # This method is used to compare two initial solutions to determine which of
        # the two is fitter. 
        if other.get_fitness() < self.fitness:
            return 1
        elif other.get_fitness() > self.fitness:
            return -1
        else:
            return 0

    # Methods in addition to the abstract methods that need to be implemented.
    def create_solution(self, trainingItems):
        # This method creates a solution using the heuristic combination.
        # Construct a solution to the problem using the heuristic combination.
        temp = ["This", " is", " a", " solution", " created", " using ", self.heuristic_combination]
        self.initial_solution = temp
        print(temp)

        student_model = trainingItems[4]
        test_dl = trainingItems[3]

        student_chk_path = "../../../NeuralNetwork/resnet20.ckpt"
        if path.exists(student_chk_path):
            student_model.load_state_dict(torch.load("../../../NeuralNetwork/resnet20.ckpt"))
        else:
            print("Path for student model weights does not exist.")
            exit()

        torch.cuda.empty_cache()

        result_before_distill = evaluate(student_model, test_dl)

        train_model_with_distillation_only(3, heuristicString=self.heuristic_combination,
                                           heuristicToLayerDict=trainingItems[0],
                                           train_dl=trainingItems[2],
                                           test_dl=trainingItems[3],
                                           student_model=trainingItems[4],
                                           student_model_number=trainingItems[5],
                                           teacher_model=trainingItems[6],
                                           teacher_model_number=trainingItems[7],
                                           device=trainingItems[8],
                                           kd_loss_type=trainingItems[13],
                                           distill_optimizer=trainingItems[14],
                                           distill_lr=trainingItems[15])

        result_after_distill = evaluate(student_model, test_dl)

        acc_change = result_after_distill['val_acc'] - result_before_distill['val_acc']
        loss_change = result_after_distill['val_loss'] - result_before_distill['val_loss']

        fitness = loss_change - acc_change

        print(acc_change)

        self.fitness = fitness

