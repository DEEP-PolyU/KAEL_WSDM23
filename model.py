from threading import Thread
from tqdm import tqdm, trange
import utils as dp
import logging
import param


class MAB():

    def __init__(self, opportunities, dictionary, detected, θ, ita) -> None:
        self.oppoetunities = opportunities
        self.dictionary = dictionary
        self.detected = detected
        self.θ = θ
        self.ita = ita

    def initialize(self, overlaps, para_lambda4):
        """
        Initialize the parameters through overlaps

        Returns:
            [θ]: [Initialized θ, 300 dimensional tensor]
            [ita]: [Initialized ita, 300 dimensional value]
        """
        print('**************Start Initialization**************')
        total_rewards = 0  # Initialize total rewards
        rewards = [[]] * param.k  # Initialize rewards
        Y = [[]] * param.k
        for ini in tqdm(range(param.k)):
            for i in range(len(overlaps)):
                #print('------ Iteration ------',i)
                reward = dp.rewarding(
                    self.dictionary, overlaps[i]
                )  # Check label by rewarding, 1 means anomaly, 0 means nominal
                rewards[ini].append(
                    reward
                )  # Update y, which indicates rewards here for current cluster
                Y[ini].append(overlaps[i])  # Update Y for current cluster
                self.θ[ini] = dp.θ_update_torch(
                    Y[ini], rewards[ini],
                    para_lambda4[ini])  # Update θ for current cluster
                self.ita[ini] = dp.ita_update_torch(
                        Y[ini], overlaps[i],
                    para_lambda4[ini])  # Update ita for current cluster
                total_rewards += reward
                #**************************************************#
        print('*************************')
        print('❀❀❀❀❀❀❀❀❀❀ Initialized successfully ❀❀❀❀❀❀❀❀❀❀')
        return self.θ, self.ita, rewards, Y

    def ranking(self, arm, triples, detected, besttriples, expectation):
        for i in range(len(triples)):
            if triples[i] not in detected:
                # Calculate the expectation
                expectationtemp = dp.calExpectation_torch(
                    triples[i], self.θ[arm], self.ita[arm])
                # Store the max expectation
                if expectationtemp >= expectation[arm]:
                    besttriples[arm] = triples[
                        i]  # Record the triple with highest expectation for current cluster
                    expectation[
                        arm] = expectationtemp  # Record the best expectation for current cluster

    def train(self, All_triples):
        global ablationcount
        global ablationrecord
        global ablationrewards
        global armchoice
        total_rewards = 0  # Initialize total rewards
        Y = [[]] * param.k
        rewards = [[] for _ in range(param.k)]  # Initialize rewards
        expectation = [0] * param.k  # Initialize Expectation
        besttriples = [None] * param.k  # Initialize the best triples for each arm
        threads = []  # Initialize the threads list

        for i in range(self.opportunities):
            print('------Oracle Iteration ------', i)
            # Create and start threads for each arm
            for arm in range(param.k):
                thread = Thread(target=self.ranking,
                                args=(arm, All_triples[arm], self.detected,
                                      besttriples, expectation))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            #**************************************************#
            bestarm = expectation.index(
                max(expectation))  # Find the best among 3
            reward = dp.rewarding(
                self.dictionary, besttriples[bestarm]
            )  # Check label by rewarding, 1 means anomaly, -0.01 means nominal
            rewards[bestarm].append(
                reward
            )  # Update y, which indicates rewards here for current cluster
            Y[bestarm].append(
                besttriples[bestarm])  # Update Y for current cluster
            self.θ[bestarm] = dp.θ_update_torch(
                Y[bestarm], rewards[bestarm],
                param.para_lambda4[bestarm])  # Update θ for current cluster
            self.ita[bestarm] = dp.ita_update_torch(
                Y[bestarm], besttriples[bestarm],
                param.para_lambda4[bestarm])  # Update ita for current cluster
            total_rewards += reward
            self.detected.append(besttriples[bestarm])

            print('***** Current TNR For arm: *****', bestarm)
            print(total_rewards / (i + 1))

        TNR = dp.calTNR(total_rewards, self.opportunities)
        return total_rewards, TNR, self.θ, self.ita, rewards, Y

    def application(self, iteration, All_triples):
        global ablationcount
        global ablationrecord
        global ablationrewards
        global armchoice
        total_rewards = 0
        besttriples = [[]] * param.k  # Initialize best choices of each cluster in in current iteration
        expectation = [0] * param.k  # Initialize Expectation
        for i in trange(iteration):
            print('------ Iteration ------', i)
            expectation = [0] * param.k
            threads = []
            for j in range(param.k):
                thread = Thread(target=self.ranking, args=(j, All_triples[j]))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
            bestarm = expectation.index(max(expectation))
            reward = dp.rewarding(self.dictionary, besttriples[bestarm])
            total_rewards += reward
            self.detected.append(besttriples[bestarm])
            print('***** Current TNR For arm: *****', bestarm)
            print(total_rewards / (i + 1))
        TNR = dp.calTNR(total_rewards, iteration)
        return total_rewards, TNR, self.θ, self.ita