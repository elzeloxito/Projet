from player_hex import PlayerHex
from seahorse.game.action import Action
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import numpy as np

class MyPlayer(PlayerHex):
    """
    Player class for Hex game

    Attributes:
        piece_type (str): piece type of the player "R" for the first player and "B" for the second player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerHex instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, name)

    # Cette fonction doit retourner une action selon l’état actuel du jeu
    # Une action est composée de deux objets GameState : (1) current_game_state : état actuel du jeu && (2) next_game_state : état du jeu après avoir effectué le mouvement proposé
    def compute_action(self, current_state: GameStateHex, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        # TODO_LIST :
        
        def heuristic(state: GameStateHex):
            return 50

        current_depth = current_state.get_step()
        limit = current_depth + 3
        player_MAX = current_state.active_player 
        id_MAX = player_MAX.get_id()
        temp_state = next(current_state.generate_possible_stateful_actions())
        min_state = temp_state.next_game_state
        player_MIN = min_state.active_player
        id_MIN = player_MIN.get_id()
        player_id = id_MAX

        states_to_extend = [current_state]
        family = {}

        # DESCENT 

        for depth in range(current_depth, limit):
            if depth != limit-1: 
                width = 5 # width should be a function of the depth but chosen as constant for the first script 
                future_states = []
                number_extended_states = 0
                found_final_states = {}

                if player_id == id_MAX:
                    for state in states_to_extend:
                        number_extended_states += 1
                        selected_states = [(None,-float('inf'))]*width
                        for extended_state in state.generate_possible_stateful_actions():
                            if not extended_state.next_game_state.is_done():
                                family[extended_state.next_game_state] = state
                                if heuristic(extended_state.next_game_state) > min(selected_states, key=lambda x: x[1])[1]:
                                    selected_states[min(range(len(selected_states)), key=lambda i: selected_states[i][1])] = (extended_state.next_game_state, heuristic(extended_state.next_game_state))
                            else: # if we find a final state during the descent, we store the value and the layer where it happened to add it to back_prop during the ascent 
                                found_final_states[extended_state.next_game_state] = ((heuristic(extended_state.next_game_state), depth))
                        future_states += [chosen[0] for chosen in selected_states if chosen[0] is not None]
                    states_to_extend = future_states
                    player_id = id_MIN

                else:
                    for state in states_to_extend:
                        number_extended_states += 1
                        selected_states = [(None,float('inf'))]*width
                        for extended_state in state.generate_possible_stateful_actions():
                            if not extended_state.next_game_state.is_done():
                                family[extended_state.next_game_state] = state
                                if heuristic(extended_state.next_game_state) < max(selected_states, key=lambda x: x[1])[1]:
                                    selected_states[max(range(len(selected_states)), key=lambda i: selected_states[i][1])] = (extended_state.next_game_state, heuristic(extended_state.next_game_state))
                            else: 
                                found_final_states[extended_state.next_game_state] = ((heuristic(extended_state.next_game_state), depth))
                        future_states += [chosen[0] for chosen in selected_states if chosen[0] is not None] 
                    states_to_extend = future_states
                    player_id = id_MAX
            
            # LAST STEP OF THE DESCENT

            else:
                back_prop = {}
                for parent_state in states_to_extend:
                    if not parent_state.is_done():
                        if player_id == id_MAX:
                            v=-99999
                            for extended_state in parent_state.generate_possible_stateful_actions():
                                if heuristic(extended_state.next_game_state)>v:
                                    v=heuristic(extended_state.next_game_state)
                            back_prop[parent_state] = v
                        
                        if player_id == id_MIN:
                            v=99999
                            for extended_state in parent_state.generate_possible_stateful_actions():
                                if heuristic(extended_state.next_game_state)<v:
                                    v=heuristic(extended_state.next_game_state)
                            back_prop[parent_state] = v
                    else:
                        back_prop[parent_state] = heuristic(parent_state)

                # ASCENT 

                for layer in range(limit - current_depth - 1):
                    if layer != limit - current_depth - 2:
                        upper_layer = {}
                        if player_id == id_MAX:
                            for child_state, value in back_prop.items():
                                if family[child_state] in upper_layer and upper_layer[family[child_state]] > value:
                                    upper_layer[family[child_state]] = value
                                else: 
                                    upper_layer[family[child_state]] = value
                            player_id = id_MIN
                        else:
                            for child_state, value in back_prop.items():
                                if family[child_state] in upper_layer and upper_layer[family[child_state]] < value:
                                    upper_layer[family[child_state]] = value
                                else: 
                                    upper_layer[family[child_state]] = value
                            player_id = id_MAX
                        
                        for found_final_state, infos in found_final_states.items():
                            if infos[1] == limit - layer - 1: 
                                back_prop[found_final_state] = infos[0]
                        back_prop = upper_layer
                        print(len(states_to_extend))

                    else:
                        v=-99999
                        for next_state, possible_value in back_prop.items():
                            if possible_value > v:
                                v=possible_value
                                chosen_state = next_state
                        for possible_action in current_state.generate_possible_stateful_actions():
                            if possible_action.next_game_state == chosen_state:
                                return possible_action
        raise MethodNotImplementedError()
