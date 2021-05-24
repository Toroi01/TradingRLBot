

class AbstractRLModel():
    """
    Here are the necessary methods that RL algorithm have to implement in order to run the backtrader
    """

    def decide_actions(self, data_row, portfolio):
        """
        Given information about the current timestamp the algorithm should decide if it should take an action
        and which.
        :param data_row: Row from the input data with the current market information
        :param portfolio: Current portfolio status
        :return: Array of Action objects or None
        """
        return None