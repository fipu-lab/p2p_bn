from abc import abstractmethod
from p2p.agents.abstract_agent import *


class SyncAgent(Agent):

    @abstractmethod
    def sync_parameters(self):
        """
            Method to be called on the end of loop
        :return:
        """
        pass

    @abstractmethod
    def update_parameters(self):
        """
            Method to be called after sync_parameters
        :return:
        """
        pass
