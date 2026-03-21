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

    def compute_action(self, current_state: GameStateHex, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """ 

        def shortest_path_computation(state: GameStateHex, board, MAX_color, RED_pieces, BLUE_pieces):

    # ------------------------------------------------ SHORTEST PATH DETECTION ------------------------------------------------

            if MAX_color == 'R':
                # BFS for shortest path between top and bottom edges
                visited_states = {}
                fringe = deque()
                top_pieces = {}
                for i in range(14):
                    top_pieces[(0,i)] = None 
                optim_len = {}
                gap_len = {}
                max_gap_path = {}

                # Start with all the top valid cells in the fringe
                for piece in top_pieces:
                    if piece in board:
                        if board[piece].get_type() == 'R':
                            optim_len[(piece, 'R')] = 0
                            gap_len[(piece, 'R')] = 0
                            max_gap_path[(piece, 'R')] = 0
                            visited_states[(piece,'R')] = (None, None) # The top pieces do not have a parent
                            fringe.appendleft((piece, 'R')) 
                    else: 
                        optim_len[(piece, 'EMPTY')] = 1
                        gap_len[(piece, 'EMPTY')] = 1
                        max_gap_path[(piece, 'EMPTY')] = 1
                        visited_states[(piece, 'EMPTY')] = (None, None) # The top pieces do not have a parent
                        fringe.append((piece, 'EMPTY'))
                
                while fringe: 
                    piece, color_type = fringe.popleft()
                    i,j = piece
                    for neighbor_infos in state.get_neighbours(i,j).values():
                        neighbor_type, neighbor_pos = neighbor_infos
                        if neighbor_type != 'OUTSIDE':
                            if neighbor_type == 'R' or neighbor_type == 'EMPTY':
                                cost = 0 if neighbor_type == 'R' else 1
                                new_cost = optim_len[(piece, color_type)] + cost

                                if ((neighbor_pos, neighbor_type) not in optim_len) or (optim_len[(neighbor_pos, neighbor_type)] > new_cost):
                                    
                                    optim_len[(neighbor_pos, neighbor_type)] = new_cost
                                    visited_states[(neighbor_pos, neighbor_type)] = (piece, color_type)

                                    # --- GAP UPDATE ---
                                    if neighbor_type == 'R':
                                        new_gap = 0
                                    else:
                                        new_gap = gap_len[(piece, color_type)] + 1

                                    gap_len[(neighbor_pos, neighbor_type)] = new_gap
                                    max_gap_path[(neighbor_pos, neighbor_type)] = max(
                                        max_gap_path[(piece, color_type)],
                                        new_gap
                                    )

                                    if neighbor_type == 'R':
                                        fringe.appendleft((neighbor_pos,neighbor_type))
                                    else:
                                        fringe.append((neighbor_pos, neighbor_type))
                
                MAX_min_cost = np.inf
                best_gap = 0

                for j in range(14):
                    for cell_type in ['R','EMPTY']:
                        key = ((13,j), cell_type)
                        if key in optim_len and optim_len[key] < MAX_min_cost:
                            MAX_min_cost = optim_len[key]
                            best_gap = max_gap_path[key]

            else:
                # BFS for shortest path between left and right edges
                visited_states = {}
                fringe = deque()
                left_pieces = {}
                for i in range(14):
                    left_pieces[(i,0)] = None 
                optim_len = {}
                gap_len = {}
                max_gap_path = {}

                # Start with all the left valid cells in the fringe
                for piece in left_pieces:
                    if piece in board:
                        if board[piece].get_type() == 'B':
                            optim_len[(piece, 'B')] = 0
                            gap_len[(piece, 'B')] = 0
                            max_gap_path[(piece, 'B')] = 0
                            visited_states[(piece,'B')] = (None, None) # The left pieces do not have a parent
                            fringe.appendleft((piece, 'B')) 
                    else: 
                        optim_len[(piece, 'EMPTY')] = 1
                        gap_len[(piece, 'EMPTY')] = 1
                        max_gap_path[(piece, 'EMPTY')] = 1
                        visited_states[(piece, 'EMPTY')] = (None, None) # The left pieces do not have a parent
                        fringe.append((piece, 'EMPTY'))
                
                while fringe: 
                    piece, color_type = fringe.popleft()
                    i,j = piece
                    for neighbor_infos in state.get_neighbours(i,j).values():
                        neighbor_type, neighbor_pos = neighbor_infos
                        if neighbor_type != 'OUTSIDE':
                            if neighbor_type == 'B' or neighbor_type == 'EMPTY':
                                cost = 0 if neighbor_type == 'B' else 1
                                new_cost = optim_len[(piece, color_type)] + cost

                                if ((neighbor_pos, neighbor_type) not in optim_len) or (optim_len[(neighbor_pos, neighbor_type)] > new_cost):
                                    
                                    optim_len[(neighbor_pos, neighbor_type)] = new_cost
                                    visited_states[(neighbor_pos, neighbor_type)] = (piece, color_type)

                                    # --- GAP UPDATE ---
                                    if neighbor_type == 'B':
                                        new_gap = 0
                                    else:
                                        new_gap = gap_len[(piece, color_type)] + 1

                                    gap_len[(neighbor_pos, neighbor_type)] = new_gap
                                    max_gap_path[(neighbor_pos, neighbor_type)] = max(
                                        max_gap_path[(piece, color_type)],
                                        new_gap
                                    )

                                    if neighbor_type == 'B':
                                        fringe.appendleft((neighbor_pos,neighbor_type))
                                    else:
                                        fringe.append((neighbor_pos, neighbor_type))
                
                MAX_min_cost = np.inf
                best_gap = 0

                for j in range(14):
                    for cell_type in ['B','EMPTY']:
                        key = ((j,13), cell_type)
                        if key in optim_len and optim_len[key] < MAX_min_cost:
                            MAX_min_cost = optim_len[key]
                            best_gap = max_gap_path[key]

            return MAX_min_cost, best_gap
        
        # Detection of different types of two_bridges 
        # The weak point is only the intercepted two-bridge 
        def two_bridge_detection(central_cell, MAX_pieces, MIN_pieces, MAX_color, board):
            i,j = central_cell
            line_out = [-1, -2, -1, 1, 2, 1]
            column_out = [-1, 1, 2, 1, -1, -2]
            line_in = [0, -1, -1, 0, 1, 1]
            column_in = [-1, 0, 1, 1, 0, -1]
            outer_crown = [(i + dx, j + dy) for dx, dy in zip(line_out, column_out)]
            inner_crown = [(i + dx, j + dy) for dx, dy in zip(line_in, column_in)]
            free_two_bridge = {}
            partially_intercepted_two_bridge = {}
            
            if MAX_color == 'R':
                for idx, cell in enumerate(outer_crown):
                    if cell in MAX_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] not in board:
                        free_two_bridge[(central_cell, cell)] = True
                    elif cell in MAX_pieces and inner_crown[idx] in MIN_pieces and inner_crown[(idx + 1) % 6] not in board:
                        partially_intercepted_two_bridge[(central_cell, cell)] = inner_crown[(idx + 1) % 6] 
                    elif cell in MAX_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] in MIN_pieces:
                        partially_intercepted_two_bridge[(central_cell, cell)] = inner_crown[idx] 
            else:
                for idx in [0, 2, 3, 5]:
                    for cell in outer_crown:
                        if cell in MAX_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] not in board:
                            free_two_bridge[(central_cell, cell)] = True
                        elif cell in MAX_pieces and inner_crown[idx] in MIN_pieces and inner_crown[(idx + 1) % 6] not in board:
                            partially_intercepted_two_bridge[(central_cell, cell)] = inner_crown[(idx + 1) % 6] 
                        elif cell in MAX_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] in MIN_pieces:
                            partially_intercepted_two_bridge[(central_cell, cell)] = inner_crown[idx] 

            return free_two_bridge, partially_intercepted_two_bridge
        

        # EDGE DETECTION 
        def close_edge_detection(MAX_pieces, MIN_pieces, MAX_color, board):

            close_to_edge_pieces = {}
            blocking_close_to_edge = {}
            confirmed_edge = {}
            edge_pieces = {}
            total_blocking = {}

            if MAX_color == 'R':
                for piece in MAX_pieces:
                    i,j = piece
                    if i == 1 and j>=0 and j+1<=13 and (0,j) in MIN_pieces and (0,j+1) in MIN_pieces:
                        total_blocking[piece] = True
            else:
                for piece in MAX_pieces:
                    i,j = piece
                    if j == 1 and i>=0 and i+1<=13 and (i,0) in MIN_pieces and (i+1,0) in MIN_pieces:
                        total_blocking[piece] = True
            
            # if MAX_color == 'R':
            #     for piece in MAX_pieces:
            #         i,j = piece
            #         if i == 1 and j>=0 and j+1<=13 and (0,j) not in board and (0,j+1) not in board:
            #             close_to_edge_pieces[piece] = True
            #         elif i == 12 and j<=13 and j-1>=0 and (13,j-1) not in board and (13,j) not in board:
            #             close_to_edge_pieces[piece] = True
            #         elif (i == 1 and j>=0 and j+1<=13 and ((0,j) in MIN_pieces or (0,j+1) in MIN_pieces)) or (i == 12 and j<=13 and j-1>=0 and ((13,j-1) in MIN_pieces or (13,j) in MIN_pieces)):
            #             blocking_close_to_edge[piece] = True
            #         elif (piece not in confirmed_edge and (0,j) not in edge_pieces) and (i == 1 and j>=0 and j+1<=13 and (0,j) in MAX_pieces):
            #             confirmed_edge[piece] = True
            #             edge_pieces[(0,j)] = True
            #         elif (piece not in confirmed_edge and (0,j+1) not in edge_pieces) and (i == 1 and j>=0 and j+1<=13 and (0,j+1) in MAX_pieces):
            #             confirmed_edge[piece] = True
            #             edge_pieces[(0,j+1)] = True
            #         elif (piece not in confirmed_edge and (13,j-1) not in edge_pieces) and (i == 12 and j<=13 and j-1>=0 and (13,j-1) in MAX_pieces):
            #             confirmed_edge[piece] = True
            #             edge_pieces[(13,j-1)] = True
            #         elif (piece not in confirmed_edge and (13,j) not in edge_pieces) and (i == 12 and j<=13 and j-1>=0 and (13,j) in MAX_pieces):
            #             confirmed_edge[piece] = True
            #             edge_pieces[(13,j)] = True
            # else:
            #     for piece in MAX_pieces:
            #         i,j = piece
            #         if j == 1 and i>=0 and i+1<=13 and (i,0) not in board and (i+1,0) not in board:
            #             close_to_edge_pieces[piece] = True
            #         elif j == 12 and i-1>=0 and i<=13 and (i,13) not in board and (i-1,13) not in board:
            #             close_to_edge_pieces[piece] = True
            #         elif (j == 1 and i>=0 and i+1<=13 and ((i,0) in MIN_pieces or (i+1,0) in MIN_pieces)) or (j == 12 and i-1>=0 and i<=13 and ((i,13) in MIN_pieces or (i-1,13) in MIN_pieces)):
            #             blocking_close_to_edge[piece] = True
            #         elif (piece not in confirmed_edge and (i,0) not in edge_pieces) and (j == 1 and i>=0 and i+1<=13 and (i,0) in MAX_pieces):
            #             confirmed_edge[piece] = True
            #             edge_pieces[(i,0)] = True
            #         elif (piece not in confirmed_edge and (i+1,0) not in edge_pieces) and (j == 1 and i>=0 and i+1<=13 and (i+1,0) in MAX_pieces):
            #             confirmed_edge[piece] = True
            #             edge_pieces[(i+1,0)] = True
            #         elif (piece not in confirmed_edge and (i,13) not in edge_pieces) and (j == 12 and i-1>=0 and i<=13 and ((i,13) in MAX_pieces)):
            #             confirmed_edge[piece] = True
            #             edge_pieces[(i,13)] = True
            #         elif (piece not in confirmed_edge and (i-1,13) not in edge_pieces) and (j == 12 and i-1>=0 and i<=13 and (i-1,13) in MAX_pieces):
            #             confirmed_edge[piece] = True
            #             edge_pieces[(i-1,13)] = True

            return total_blocking, close_to_edge_pieces, blocking_close_to_edge, confirmed_edge
        
        def potential_blocking_detection(RED_pieces, BLUE_pieces, MAX_color):

            RED_blocked_pieces = {}
            BLUE_blocked_pieces = {}

            # for piece_infos in RED_path:
            #     piece_pos, piece_type = piece_infos
            #     i,j = piece_pos
            #     # Upper blocking
            #     line_in = [0, -1, -1, 0, 1, 1]
            #     column_in = [-1, 0, 1, 1, 0, -1]
            #     inner_crown = [(i + dx, j + dy) for dx, dy in zip(line_in, column_in)]
            #     for idx, adj_cell in enumerate(inner_crown):
            #         if inner_crown[idx] in BLUE_pieces and inner_crown[(idx + 1) % 6] in BLUE_pieces:
            #             RED_blocked_pieces[piece_pos] = True

            for piece in RED_pieces:
                i,j = piece
                # Left blocking
                if (i-1,j) in BLUE_pieces and (i-1,j+1) in BLUE_pieces:
                    RED_blocked_pieces[piece] = True
                # Right blocking
                if (i+1,j-1) in BLUE_pieces and (i+1,j) in BLUE_pieces:
                    RED_blocked_pieces[piece] = True

            for piece in BLUE_pieces:
                i,j = piece
                # Left blocking
                if (i,j-1) in RED_pieces and (i+1,j-1) in RED_pieces:
                    BLUE_blocked_pieces[piece] = True
                # Right blocking
                if (i,j+1) in RED_pieces and (i-1,j+1) in RED_pieces:
                    BLUE_blocked_pieces[piece] = True

            if MAX_color == 'R':
                MAX_blocked_pieces = RED_blocked_pieces
                MIN_blocked_pieces = BLUE_blocked_pieces
            else:
                MAX_blocked_pieces = BLUE_blocked_pieces
                MIN_blocked_pieces = RED_blocked_pieces

            return MAX_blocked_pieces, MIN_blocked_pieces
    
        # ------------------------------------------------ HEURISTIC ------------------------------------------------
        def heuristic(state: GameStateHex):

            MAX_color = self.piece_type
            board = state.get_rep().get_env()
            MAX_pieces = {}
            MIN_pieces = {}
            free_two_bridges = {}
            partially_intercepted_two_bridges = {}
            h_value = 0

            for piece, piece_type in board.items():
                if piece_type.get_owner_id() == id_MAX:
                    MAX_pieces[piece] = True
                else:
                    MIN_pieces[piece] = True
            
            if MAX_color == 'R':
                RED_pieces = MAX_pieces
                BLUE_pieces = MIN_pieces
            else:
                RED_pieces = MIN_pieces
                BLUE_pieces = MAX_pieces

            if not state.is_done():
                # ---------- GOOD GENERAL CONNECTIVITY ----------
                # TWO BRIDGES DETECTION
                for piece in MAX_pieces:
                    two_bridges, int_two_bridges = two_bridge_detection(piece, MAX_pieces, MIN_pieces, MAX_color, board)
                    for bridge in two_bridges:
                        rev_bridge = bridge[::-1]
                        if bridge not in free_two_bridges and rev_bridge not in free_two_bridges:
                            free_two_bridges[bridge] = True
                    for int_bridge in int_two_bridges:
                        rev_int_bridge = int_bridge[::-1]
                        if int_bridge not in partially_intercepted_two_bridges and rev_int_bridge not in partially_intercepted_two_bridges:
                            partially_intercepted_two_bridges[int_bridge] = True
            
                n_free_two_bridges = len(free_two_bridges)
                n_int_two_bridges = len(partially_intercepted_two_bridges)
                w_free_two_bridges = 8
                w_int_two_bridges = 25
                h_value += n_free_two_bridges*w_free_two_bridges - n_int_two_bridges*w_int_two_bridges

                # ---------- ORIENTATION ----------
                MAX_optimal_length, max_gap = shortest_path_computation(state, board, MAX_color, RED_pieces, BLUE_pieces)
                w_len_MAX_path = 20
                if depth <= 14:
                    w_max_gap = 5
                else:
                    w_max_gap = 0

                h_value -= (MAX_optimal_length*w_len_MAX_path  + max_gap*w_max_gap)

                # ---------- EDGE CONNECTION ----------
                # Close edge 
                MAX_total_blocking, MAX_close_edges, MAX_blocking_close_edge, MAX_confirmed_edges = close_edge_detection(MAX_pieces, MIN_pieces, MAX_color, board)
                n_close_edges = len(MAX_close_edges)
                n_blocking_close_edge = len(MAX_blocking_close_edge)
                n_confirmed_edges = len(MAX_confirmed_edges)
                n_total_blocking = len(MAX_total_blocking)
                w_close_edges = 0
                w_blocking_close_edge = 20
                w_confirmed_edges = 5
                w_total_blocking = 20

                h_value += (n_close_edges*w_close_edges + n_confirmed_edges*w_confirmed_edges - n_blocking_close_edge*w_blocking_close_edge - n_total_blocking*w_total_blocking)

                # ---------- BLOCKING ----------
                MAX_blocked_pieces, MIN_blocked_pieces = potential_blocking_detection(RED_pieces, BLUE_pieces, MAX_color)
                n_MAX_blocked_pieces = len(MAX_blocked_pieces)
                n_MIN_blocked_pieces = len(MIN_blocked_pieces)
                w_MAX_blocked_pieces = 70
                w_MIN_blocked_pieces = 20

                h_value += (n_MIN_blocked_pieces*w_MIN_blocked_pieces - n_MAX_blocked_pieces*w_MAX_blocked_pieces)
            
            else:
                if state.get_scores()[id_MAX] == 1.0:
                    h_value = 1e6
                else:
                    h_value = -1e6

            return h_value
        
        # ------------------------------------------------ MIN/MAX - ALPHA/BETA - ALGORITHM ------------------------------------------------
        # For the advanced moves

        board = current_state.get_rep().get_env()

        depth = current_state.get_step()
        if depth <= 14:
            limit_depth = depth + 3
            width = 5
        else: 
            limit_depth = depth + 5
            width = 5
        id_MAX = current_state.active_player.get_id()
        first_moves = list(range(4))

        if depth == 1:
            first_MIN_piece = next(iter(board))

        first_MAX_piece = ()
        if depth == 3:
            for piece, piece_type in board.items():
                if piece_type != id_MAX:
                    first_MAX_piece = piece

        if depth not in first_moves:

            def player_MAX(current_state, alpha, beta, depth):
                if current_state.is_done() and current_state.get_scores()[id_MAX] == 1.0:
                    return (1e6, None)
                elif current_state.is_done() and current_state.get_scores()[id_MAX] == 0.0:
                    return(-1e6, None)
                elif depth == limit_depth:
                    return (heuristic(current_state), None)

                best_estimation = -np.inf
                best_action = None
                all_actions = list(current_state.generate_possible_stateful_actions())
                number_of_actions = len(all_actions)
                if number_of_actions >= width:
                    selected_actions = [(None, -np.inf) for _ in range(width)]
                else: 
                    selected_actions = [(None, -np.inf) for _ in range(number_of_actions)]
                for action in all_actions:
                    next_state = action.next_game_state
                    if next_state.is_done() and next_state.get_scores()[id_MAX] == 1.0:
                        return (1e6, action)
                    idx, value = min(enumerate(selected_actions), key=lambda x: x[1][1])
                    h_value = heuristic(next_state)
                    # print(h_value)
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
                if current_state.is_done() and current_state.get_scores()[id_MAX] == 1.0:
                    return (1e6, None)
                elif current_state.is_done() and current_state.get_scores()[id_MAX] == 0.0:
                    return(-1e6, None)
                elif depth == limit_depth:
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
                    # print(h_value)
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
        
        # ------------------------------------------------ OPENING MOVES ------------------------------------------------
        # For the first moves 
        else:
            MAX_color = self.piece_type
            center_pieces = []
            for i in range(5,9):
                for j in range(5,9):
                    center_pieces.append((i,j))

            if depth == 0: # => player MAX is a RED player - first move
                for action in current_state.generate_possible_stateful_actions():
                    if (10,3) in action.get_next_game_state().get_rep().get_env():
                        return action
                    
            elif depth == 1: # => player MAX is a BLUE player - first move
                for action in current_state.generate_possible_stateful_actions():
                    if (10,3) in action.get_next_game_state().get_rep().get_env():
                        return action

            elif depth == 2: # => player MAX is a RED player - second move
                MIN_piece = next(iter(board))
                upper_part = range(7)
                i,j = MIN_piece
                if MIN_piece != (3,10):
                    for action in current_state.generate_possible_stateful_actions():
                                if (3,10) in action.get_next_game_state().get_rep().get_env():
                                    return action
                else:
                    for action in current_state.generate_possible_stateful_actions():
                                if (2,4) in action.get_next_game_state().get_rep().get_env():
                                    return action
            
            elif depth == 3:
                for action in current_state.generate_possible_stateful_actions():
                    if (3,10) in action.get_next_game_state().get_rep().get_env():
                        return action

        raise MethodNotImplementedError()
