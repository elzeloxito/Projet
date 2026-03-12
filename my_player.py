from player_hex import PlayerHex
from seahorse.game.action import Action
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import numpy as np
from collections import deque

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
    def compute_action(self, current_state: GameStateHex, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """ 

        def heuristic(state: GameStateHex):     
            
            # Size, diameter, orientation and distance to the edges of different groups 
            board = state.get_rep().get_env()
            MAX_color = self.piece_type # R or B
            shortest_path = np.inf 

            if MAX_color == 'R': # if MAX_player is RED => edges are at the top and bottom of the board
                # BFS for shortest path between top and bottom edges
                visited_states = {}
                fringe = deque()
                board = state.get_rep().get_env()
                top_pieces = []
                for i in range(14):
                    top_pieces.append((0,i))
                MAX_pieces = {}
                MIN_pieces = {}
                shortest_path = np.inf
                found_edge = 0
                # start with all the top valid pieces in the fringe
                for piece in top_pieces:
                    if piece in board:
                        if board[piece].get_type() == MAX_color:
                            visited_states[piece] = True
                            fringe.appendleft((piece, 0)) # initial score of 0 for played pieces 
                    else: 
                        visited_states[piece] = True
                        fringe.append((piece, 1))
                
                for piece in board:
                    if board[piece].get_type() == MAX_color:
                        MAX_pieces[piece] = True
                    else:
                        MIN_pieces[piece] = True
                
                while fringe: 
                    piece, score = fringe.popleft()
                    i,j = piece
                    for neighbor in state.get_neighbours(i,j).values():
                        # Case the neighbor is an empty cell
                        neighbor = neighbor[1]
                        if neighbor not in visited_states and neighbor not in MAX_pieces and neighbor not in MIN_pieces:
                            if neighbor[0] == 13:
                                shortest_path = score + 1
                                found_edge = 1
                                break
                            else:
                                fringe.append((neighbor, score + 1))
                                visited_states[neighbor] = True
                        # Case the neighbor is a played piece (cell with a RED piece)
                        elif neighbor not in visited_states and neighbor in MAX_pieces and neighbor not in MIN_pieces:
                            if neighbor[0] == 13:
                                shortest_path = score
                                found_edge = 1
                                break
                            else:
                                fringe.appendleft((neighbor, score))
                                visited_states[neighbor] = True 
                    if found_edge == 1:
                        break
            
            else: # if MAX_player is BLUE => edges are at the left and the right of the board
                # BFS for shortest path between left and right edges
                visited_states = {}
                fringe = deque()
                board = state.get_rep().get_env()
                top_pieces = []
                for i in range(14):
                    top_pieces.append((i,0))
                MAX_pieces = {}
                MIN_pieces = {}
                shortest_path = np.inf
                found_edge = 0
                # start with all the top valid pieces in the fringe
                for piece in top_pieces:
                    if piece in board:
                        if board[piece].get_type() == MAX_color:
                            visited_states[piece] = True
                            fringe.appendleft((piece, 0)) # initial score of 0 for played pieces 
                    else: 
                        visited_states[piece] = True
                        fringe.append((piece, 1))
                
                for piece in board:
                    if board[piece].get_type() == MAX_color:
                        MAX_pieces[piece] = True
                    else:
                        MIN_pieces[piece] = True
                
                while fringe: 
                    piece, score = fringe.popleft()
                    i,j = piece
                    for neighbor in state.get_neighbours(i,j).values():
                        # Case the neighbor is an empty cell
                        neighbor = neighbor[1]
                        if neighbor not in visited_states and neighbor not in MAX_pieces and neighbor not in MIN_pieces:
                            if neighbor[1] == 13:
                                shortest_path = score + 1
                                found_edge = 1
                                break
                            else:
                                fringe.append((neighbor, score + 1))
                                visited_states[neighbor] = True
                        # Case the neighbor is a played piece (cell with a RED piece)
                        elif neighbor not in visited_states and neighbor in MAX_pieces and neighbor not in MIN_pieces:
                            if neighbor[1] == 13:
                                shortest_path = score
                                found_edge = 1
                                break
                            else:
                                fringe.appendleft((neighbor, score))
                                visited_states[neighbor] = True 
                    if found_edge == 1:
                        break
            
            if shortest_path !=0:
                return 1/shortest_path
            else: 
                return np.inf
                
        depth = current_state.get_step()
        limit_depth = depth + 3
        width = 5

        if current_state.get_step() !=0:

            def player_MAX(current_state, alpha, beta, depth):
                if current_state.is_done() or depth == limit_depth:
                    return (heuristic(current_state), None)

                best_estimation = -np.inf
                best_action = None
                number_of_actions = len(list(current_state.generate_possible_stateful_actions()))
                if number_of_actions >= width:
                    selected_actions = [(None, -np.inf) for _ in range(width)]
                else: 
                    selected_actions = [(None, -np.inf) for _ in range(number_of_actions)]
                for action in current_state.generate_possible_stateful_actions():
                    idx, value = min(enumerate(selected_actions), key=lambda x: x[1][1])
                    h_value = heuristic(action.next_game_state)
                    print(h_value)
                    if  h_value > value[1]:
                        selected_actions[idx] = (action, h_value)
                for action, h_value in selected_actions:
                    next_state = action.next_game_state
                    next_estimation, _ = player_MIN(next_state, alpha, beta, depth + 1)
                    if next_estimation > best_estimation:
                        best_estimation = next_estimation
                        best_action = action
                        alpha = max(alpha, best_estimation)
                    if best_estimation >= beta:
                        return (best_estimation, best_action)
                return (best_estimation, best_action)

            def player_MIN(current_state, alpha, beta, depth):
                if current_state.is_done() or depth == limit_depth:
                    return (heuristic(current_state), None)

                best_estimation = np.inf
                best_action = None
                number_of_actions = len(list(current_state.generate_possible_stateful_actions()))
                if number_of_actions >= width:
                    selected_actions = [(None, np.inf) for _ in range(width)]
                else:
                    selected_actions = [(None, np.inf) for _ in range(number_of_actions)]
                for action in current_state.generate_possible_stateful_actions():
                    idx, value = max(enumerate(selected_actions), key=lambda x: x[1][1])
                    h_value = heuristic(action.next_game_state)
                    print(h_value)
                    if  h_value < value[1]:
                        selected_actions[idx] = (action, h_value)
                for action, h_value in selected_actions:
                    next_state = action.next_game_state
                    next_estimation, _ = player_MAX(next_state, alpha, beta, depth + 1)
                    if next_estimation < best_estimation:
                        best_estimation = next_estimation
                        best_action = action
                        beta = min(beta, best_estimation)
                    if best_estimation <= alpha:
                        return (best_estimation, best_action)
                return (best_estimation, best_action)
            
            return player_MAX(current_state, -np.inf, np.inf, depth)[1]
        else:
            for action in current_state.generate_possible_stateful_actions():
                if (7,7) in action.get_next_game_state().get_rep().get_env():
                    return action
        raise MethodNotImplementedError()
