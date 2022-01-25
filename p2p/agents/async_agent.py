from abc import abstractmethod
from p2p.agents.abstract_agent import *


class AsyncAgent(Agent):

    @abstractmethod
    def can_be_awaken(self):
        """
            Method to be called to check whether an agent can be chosen to train
        :return:
        """
        pass
