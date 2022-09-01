#!/usr/bin/env python
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
import torch

import sys

sys.path.append('/mnt/lustre/users/btomkinson/distill_code')

from GeneticAlgorithm.GeneticAlgorithm import GeneticAlgorithm
from GeneticAlgorithm.distillation.DistillationProblem import DistillationProblem


class GeneticAlgorithmMain(object):
    @classmethod
    def solve(cls):
        # This method illustrates how the selection construction hyper-heuristic in
        # the GeneticAlgorithm library can be used to solve a combinatorial optimization problem.
        # abcdefghijklmnopqrstuvwxyz1234567890
        # abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ12
        problem = DistillationProblem()
        seed = round(time.time() * 1000)
        heuristics = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ12"
        genetic_algorithm = GeneticAlgorithm(seed, heuristics)
        genetic_algorithm.set_parameters("Parameters.txt")
        genetic_algorithm.set_problem(problem)

        teacher_model_number = 18  # ResNet 110
        student_model_number = 9  # ResNet 56   3 = ResNet 20

        initial_epochs = 10
        epochs = 70
        total_epochs = 80
        BATCH_SIZE = 100

        optimizer = torch.optim.Adam
        scheduler = torch.optim.lr_scheduler.OneCycleLR
        max_lr = 0.003
        grad_clip = 0.1
        weight_decay = 0

        distill_optimizer = torch.optim.Adam
        distill_lr = 0.0003
        kd_loss_type = 'cosine'
        numOfBatchesToDistill = 2

        heuristicToLayerDict = {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 4,
            'e': 5,
            'f': 6,
            'g': 7,
            'h': 8,
            'i': 9,
            'j': 10,
            'k': 11,
            'l': 12,
            'm': 13,
            'n': 14,
            'o': 15,
            'p': 16,
            'q': 17,
            'r': 18,
            's': 19,
            't': 20,
            'u': 21,
            'v': 22,
            'w': 23,
            'x': 24,
            'y': 25,
            'z': 26,
            '1': 27,
            '2': 28,
            '3': 29,
            '4': 30,
            '5': 31,
            '6': 32,
            '7': 33,
            '8': 34,
            '9': 35,
            '0': 36,
        }

        heuristicToLayerDict56 = {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 4,
            'e': 5,
            'f': 6,
            'g': 7,
            'h': 8,
            'i': 9,
            'j': 10,
            'k': 11,
            'l': 12,
            'm': 13,
            'n': 14,
            'o': 15,
            'p': 16,
            'q': 17,
            'r': 18,
            's': 19,
            't': 20,
            'u': 21,
            'v': 22,
            'w': 23,
            'x': 24,
            'y': 25,
            'z': 26,
            'A': 27,
            'B': 28,
            'C': 29,
            'D': 30,
            'E': 31,
            'F': 32,
            'G': 33,
            'H': 34,
            'I': 35,
            'J': 36,
            'K': 37,
            'L': 38,
            'M': 39,
            'N': 40,
            'O': 41,
            'P': 42,
            'Q': 43,
            'R': 44,
            'S': 45,
            'T': 46,
            'U': 47,
            'V': 48,
            'W': 49,
            'X': 50,
            'Y': 51,
            'Z': 52,
            '1': 53,
            '2': 54,
        }

        print("HYPER PARAMETERS:")
        print("Number of initial epochs: " + str(initial_epochs))
        print("Number of total epochs: " + str(total_epochs))
        print("Number of epochs: " + str(epochs))
        print("Batch size: " + str(BATCH_SIZE))
        print("Optimizer: " + str(optimizer))
        print("Max learning rate: " + str(max_lr))
        print("Gradient clip value: " + str(grad_clip))
        print("Weight decay: " + str(weight_decay))
        print("Scheduler: " + str(scheduler))
        print("KD loss type: " + str(kd_loss_type))
        print("Distill  optimizer : " + str(distill_optimizer))
        print("Distill  optimizer learning rate: " + str(distill_lr))
        print("Batches distilled per epoch: " + str(numOfBatchesToDistill))

        trainingParameters = [teacher_model_number, student_model_number, BATCH_SIZE, epochs, optimizer, max_lr,
                              distill_optimizer, distill_lr, grad_clip, weight_decay, scheduler, kd_loss_type,
                              heuristicToLayerDict56, numOfBatchesToDistill, initial_epochs, total_epochs]
        solution = genetic_algorithm.evolve(trainingParameters)

        print("Best Solution")
        print("--------------")
        print("Fitness:", solution.get_fitness())
        print("Heuristic combination: " + solution.get_heuristic_combination())
        print("Solution: ")
        GeneticAlgorithmMain.display_solution(solution.get_solution())
        # Print NN parameters:
        print("HYPER PARAMETERS:")
        print("Number of initial epochs: " + str(initial_epochs))
        print("Number of total epochs: " + str(total_epochs))
        print("Number of epochs: " + str(epochs))
        print("Batch size: " + str(BATCH_SIZE))
        print("Optimizer: " + str(optimizer))
        print("Max learning rate: " + str(max_lr))
        print("Gradient clip value: " + str(grad_clip))
        print("Weight decay: " + str(weight_decay))
        print("Scheduler: " + str(scheduler))
        print("KD loss type: " + str(kd_loss_type))
        print("Distill  optimizer : " + str(distill_optimizer))
        print("Distill  optimizer learning rate: " + str(distill_lr))
        print("Batches distilled per epoch: " + str(numOfBatchesToDistill))

    @classmethod
    def display_solution(cls, solution):
        # Displays the solution.
        print(' '.join(solution))

    @classmethod
    def main(cls):
        cls.solve()


if __name__ == '__main__':
    GeneticAlgorithmMain.main()
