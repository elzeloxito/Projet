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

        def shortest_path_computation(state: GameStateHex):

            board = state.get_rep().get_env()
            MAX_color = self.piece_type
            RED_pieces = {}
            BLUE_pieces = {}
            MAX_pieces = {}
            MIN_pieces = {}

            for piece in board:
                if board[piece].get_type() == 'R':
                    RED_pieces[piece] = True
                else:
                    BLUE_pieces[piece] = True 
            
            if MAX_color == 'R':
                MAX_pieces = RED_pieces
                MIN_pieces = BLUE_pieces
            else:
                MAX_pieces = BLUE_pieces
                MIN_pieces = RED_pieces

            # ------------------------------------------------ SHORTEST PATH DETECTION ------------------------------------------------
            # BFS for shortest path between top and bottom edges
            visited_states = {}
            fringe = deque()
            top_pieces = {}
            for i in range(5,9):
                top_pieces[(0,i)] = None 
            for i in range(5):
                top_pieces[(0,i)] = None 
            for i in range(9,14):
                top_pieces[(0,i)] = None 
            found_edge = 0

            # Start with all the top valid cells in the fringe
            for piece in top_pieces:
                if piece in board:
                    if board[piece].get_type() == 'R':
                        visited_states[(piece,'R')] = None # The top pieces do not have a parent
                        fringe.appendleft((piece, 'R')) 
                else: 
                    visited_states[(piece, 'EMPTY')] = None # The top pieces do not have a parent
                    fringe.append((piece, 'EMPTY'))
            
            while fringe: 
                piece, color_type = fringe.popleft()
                i,j = piece
                for _, neighbor_pos in state.get_neighbours(i,j).values():
                    if neighbor_pos == 'OUTSIDE':
                        continue
                    if not (0 <= neighbor_pos[0] <= 13 and 0 <= neighbor_pos[1] <= 13):
                        continue
                    if neighbor_pos in board:
                        neighbor_type = board[neighbor_pos].get_type()
                    else:
                        neighbor_type = 'EMPTY'
                    # Case the neighbor is an empty cell
                    if (neighbor_pos, neighbor_type) not in visited_states and neighbor_pos not in board and neighbor_type != 'OUTSIDE':
                        if neighbor_pos[0] == 13:
                            found_edge = 1
                            back_prop_state = (neighbor_pos, neighbor_type)
                            visited_states[(neighbor_pos, neighbor_type)] = (piece, color_type)
                            break
                        else:
                            fringe.append((neighbor_pos, neighbor_type))
                            visited_states[(neighbor_pos, neighbor_type)] = (piece, color_type)
                    # Case the neighbor is a played piece (cell with a RED piece)
                    elif (neighbor_pos, neighbor_type) not in visited_states and neighbor_pos in RED_pieces and neighbor_type != 'OUTSIDE':
                        if neighbor_pos[0] == 13:
                            found_edge = 1
                            back_prop_state = (neighbor_pos, neighbor_type)
                            visited_states[(neighbor_pos, neighbor_type)] = (piece, color_type)
                            break
                        else:
                            fringe.appendleft((neighbor_pos, neighbor_type))
                            visited_states[(neighbor_pos, neighbor_type)] = (piece, color_type)  
                if found_edge == 1:
                    break
            
            pieces_vertical_path = []
            while back_prop_state[0] not in top_pieces:
                pieces_vertical_path.append(back_prop_state)
                back_prop_state = visited_states[back_prop_state]
            pieces_vertical_path.append(back_prop_state)

            
            # BFS for shortest path between left and right edges
            visited_states = {}
            fringe = deque()
            left_pieces = {}
            for i in range(14):
                left_pieces[(i,0)] = None 
            found_edge = 0
            # start with all the left valid cells in the fringe
            for piece in left_pieces:
                if piece in board:
                    if board[piece].get_type() == 'B':
                        visited_states[(piece,'B')] = None # The left pieces do not have a parent
                        fringe.appendleft((piece, 'B')) 
                else: 
                    visited_states[(piece, 'EMPTY')] = None # The left p pieces do not have a parent
                    fringe.append((piece, 'EMPTY'))

            while fringe: 
                piece, color_type = fringe.popleft()
                i,j = piece
                for _, neighbor_pos in state.get_neighbours(i,j).values():
                    if neighbor_pos == 'OUTSIDE':
                        continue
                    if not (0 <= neighbor_pos[0] <= 13 and 0 <= neighbor_pos[1] <= 13):
                        continue
                    if neighbor_pos in board:
                        neighbor_type = board[neighbor_pos].get_type()
                    else:
                        neighbor_type = 'EMPTY'
                    # Case the neighbor is an empty cell
                    if (neighbor_pos, neighbor_type) not in visited_states and neighbor_pos not in board and neighbor_type != 'OUTSIDE':
                        if neighbor_pos[1] == 13:
                            found_edge = 1
                            back_prop_state = (neighbor_pos, neighbor_type)
                            visited_states[(neighbor_pos, neighbor_type)] = (piece, color_type)
                            break
                        else:
                            fringe.append((neighbor_pos, neighbor_type))
                            visited_states[(neighbor_pos, neighbor_type)] = (piece, color_type)
                    # Case the neighbor is a played piece (cell with a BLUE piece)
                    elif (neighbor_pos, neighbor_type) not in visited_states and neighbor_pos in BLUE_pieces and neighbor_type != 'OUTSIDE':
                        if neighbor_pos[1] == 13:
                            found_edge = 1
                            back_prop_state = (neighbor_pos, neighbor_type)
                            visited_states[(neighbor_pos, neighbor_type)] = (piece, color_type)
                            break
                        else:
                            fringe.appendleft((neighbor_pos, neighbor_type))
                            visited_states[(neighbor_pos, neighbor_type)] = (piece, color_type)  
                if found_edge == 1:
                    break
            
            pieces_horizontal_path = []
            while back_prop_state[0] not in left_pieces:
                pieces_horizontal_path.append(back_prop_state)
                back_prop_state = visited_states[back_prop_state]
            pieces_horizontal_path.append(back_prop_state)
            
            return pieces_vertical_path, pieces_horizontal_path, MAX_pieces, MIN_pieces

        def key_points_identification (state: GameStateHex, pieces_vertical_path, pieces_horizontal_path, MAX_pieces, MIN_pieces):
            
            MAX_color = self.piece_type # R or B
            board = state.get_rep().get_env()

            # ------------------------------------------------ WEAK POINT DETECTION ------------------------------------------------
            # Detection of different types of two_bridges 
            # The weak point is only the intercepted two-bridge 

            def outer_crown_computation(central_cell):
                i,j = central_cell
                line_out = [-1, -2, -1, 1, 2, 1]
                column_out = [-1, 1, 2, 1, -1, -2]
                outer_crown = [(i + dx, j + dy) for dx, dy in zip(line_out, column_out)]
                return outer_crown
            
            def inner_crown_computation(central_cell):
                i,j = central_cell
                line_in = [0, -1, -1, 0, 1, 1]
                column_in = [-1, 0, 1, 1, 0, -1]
                inner_crown = [(i + dx, j + dy) for dx, dy in zip(line_in, column_in)]
                return inner_crown

            def two_bridge_detection():

                # TWO BRIDGE DETECTION
                # MAX case 
                # POTENTIAL FREE TWO BRIDGES
                MAX_potential_two_bridges = {}
                MAX_confirmed_two_bridges = {}

                if MAX_pieces:
                    for central_cell in MAX_pieces:
                        potential_two_bridges = []
                        confirmed_two_bridges = []
                        outer_crown = outer_crown_computation(central_cell)
                        inner_crown = inner_crown_computation(central_cell)
                        idx = 0
                        for idx, out_piece in enumerate(outer_crown):
                            if inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] not in board:
                                potential_two_bridges.append(out_piece)
                            if out_piece in MAX_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] not in board:
                                confirmed_two_bridges.append(out_piece)
                        if potential_two_bridges:
                            MAX_potential_two_bridges[central_cell] = potential_two_bridges
                        if confirmed_two_bridges:
                            MAX_confirmed_two_bridges[central_cell] = confirmed_two_bridges

                # MIN case
                # CONFIRMED FREE TWO BRIDGES 
                MIN_confirmed_two_bridges = {}
                if MIN_pieces:
                    for central_cell in MIN_pieces:
                        potential_two_bridges = []
                        outer_crown = outer_crown_computation(central_cell)
                        inner_crown = inner_crown_computation(central_cell)
                        idx = 0
                        for idx, out_piece in enumerate(outer_crown):
                            if out_piece in MIN_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] not in board:
                                potential_two_bridges.append(out_piece)
                        if potential_two_bridges:
                            MIN_confirmed_two_bridges[central_cell] = potential_two_bridges

                return MAX_confirmed_two_bridges, MAX_potential_two_bridges, MIN_confirmed_two_bridges
        
            def partially_intercepted_two_bridges():

                # PARTIALLY INTERCEPTED TWO BRIDGES DETECTION
                # MAX case
                MAX_partially_intercepted_two_bridge = {}
                if MAX_pieces:
                    for central_cell in MAX_pieces:
                        outer_crown = outer_crown_computation(central_cell)
                        inner_crown = inner_crown_computation(central_cell)
                        for idx, cell in enumerate(outer_crown):
                            if cell in MAX_pieces and inner_crown[idx] in MIN_pieces and inner_crown[(idx + 1) % 6] not in board:
                                MAX_partially_intercepted_two_bridge[(central_cell, cell)] = inner_crown[(idx + 1) % 6] # We store the two-bridge pieces and the empty cell between the two
                            elif cell in MAX_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] in MIN_pieces:
                                MAX_partially_intercepted_two_bridge[(central_cell, cell)] = inner_crown[idx] # We store the two-bridge pieces and the empty cell between the two
                            
                # MIN case
                MIN_partially_intercepted_two_bridge = {}
                MIN_potential_partially_intercepted_two_bridge = {}
                if MIN_pieces:
                    for central_cell in MIN_pieces:
                        outer_crown = outer_crown_computation(central_cell)
                        inner_crown = inner_crown_computation(central_cell)
                        potential_intercepted_two_bridges = []
                        for idx, out_piece in enumerate(outer_crown):
                            if out_piece in MIN_pieces and inner_crown[idx] in MAX_pieces and inner_crown[(idx + 1) % 6] not in board:
                                MIN_partially_intercepted_two_bridge[(central_cell, out_piece)] = inner_crown[(idx + 1) % 6] # We store the two-bridge pieces and the empty cell between the two
                            elif out_piece in MIN_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] in MAX_pieces:
                                MIN_partially_intercepted_two_bridge[(central_cell, out_piece)] = inner_crown[idx] # We store the two-bridge pieces and the empty cell between the two
                        
                            if out_piece in MIN_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] not in board:
                                if inner_crown[idx] not in potential_intercepted_two_bridges:
                                    potential_intercepted_two_bridges.append(inner_crown[idx])
                                if inner_crown[(idx + 1) % 6] not in potential_intercepted_two_bridges:
                                    potential_intercepted_two_bridges.append(inner_crown[(idx + 1) % 6])
                        if potential_intercepted_two_bridges:
                            MIN_potential_partially_intercepted_two_bridge[central_cell] = potential_intercepted_two_bridges
                
                return MAX_partially_intercepted_two_bridge, MIN_partially_intercepted_two_bridge, MIN_potential_partially_intercepted_two_bridge

            # Detection of a potential blocking move from player MIN
            # Weak points are states that will lead to a block from player MIN if no reaction from player MAX

            def possible_blocking(pieces_vertical_path:list, pieces_horizontal_path:list):

                red_possible_blocking = {}
                red_potential_blocking = {}
                blue_possible_blocking = {}
                blue_potential_blocking = {}
                red_pieces = [c for c in pieces_vertical_path if c[1] == 'R']
                blue_pieces = [c for c in pieces_horizontal_path if c[1] == 'B']

                for red_idx, red_piece in enumerate(red_pieces):
                    if red_idx == len(red_pieces) - 1:
                        continue
                    else:
                        i_red_1, j_red_1 = red_pieces[red_idx][0]
                        i_red_2, j_red_2 = red_pieces[red_idx + 1][0]
                        gap = abs(i_red_2 - i_red_1)
                        for blue_idx, blue_piece in enumerate(blue_pieces):
                            blue_i, blue_j = blue_pieces[blue_idx][0]
                            
                            if min(i_red_1, i_red_2) < blue_i < max(i_red_1, i_red_2):
                                if min(j_red_1, j_red_2) <= blue_j <= max(j_red_1, j_red_2):
                                    if gap > 2:
                                        defensive_moves = [(blue_i - 2, blue_j), (blue_i - 2, blue_j + 1), (blue_i - 2, blue_j + 2),
                                                        (blue_i + 2, blue_j - 1), (blue_i + 2, blue_j), (blue_i + 1, blue_j + 1)]
                                    else:
                                        defensive_moves = [(blue_i - 1, blue_j), (blue_i - 1, blue_j - 1), (blue_i - 2, blue_j - 1),
                                                        (blue_i + 1, blue_j), (blue_i + 1, blue_j - 1), (blue_i + 2, blue_j - 1)]
                                    red_possible_blocking[blue_piece] = defensive_moves
                                    # Potentiel bloc = les mêmes cellules vues comme cibles d'attaque pour BLUE
                                    blue_potential_blocking[blue_piece] = defensive_moves

                            else:
                                misalignment = min(abs(blue_j - j_red_1), abs(blue_j - j_red_2))
                                if misalignment < 3:
                                    a = 1 if blue_j < min(j_red_1, j_red_2) else -1
                                    defensive_moves = [(blue_i, blue_j + a*misalignment + a), (blue_i, blue_j + a*misalignment + 2*a),
                                                    (blue_i - a, blue_j + a*misalignment + a), (blue_i - a, blue_j + a*misalignment + 2*a),
                                                    (blue_i - 2*a, blue_j + a*misalignment + a), (blue_i - 2*a, blue_j + a*misalignment + 2*a)]
                                    red_possible_blocking[blue_piece] = defensive_moves
                                    blue_potential_blocking[blue_piece] = defensive_moves

                for blue_idx, blue_piece in enumerate(blue_pieces):
                    if blue_idx == len(blue_pieces) - 1:
                        continue
                    else:
                        i_blue_1, j_blue_1 = blue_pieces[blue_idx][0]
                        i_blue_2, j_blue_2 = blue_pieces[blue_idx + 1][0]
                        gap = abs(j_blue_2 - j_blue_1)
                        for red_idx, red_piece in enumerate(red_pieces):
                            red_i, red_j = red_pieces[red_idx][0]
                            
                            if min(j_blue_1, j_blue_2) < red_j < max(j_blue_1, j_blue_2):
                                if min(i_blue_1, i_blue_2) <= red_i <= max(i_blue_1, i_blue_2):
                                    if gap > 2:
                                        defensive_moves = [(red_i, red_j - 2), (red_i + 1, red_j - 2), (red_i + 2, red_j - 2),
                                                        (red_i - 1, red_j + 2), (red_i, red_j + 2), (red_i + 1, red_j + 1)]
                                    else:
                                        defensive_moves = [(red_i, red_j - 1), (red_i - 1, red_j - 1), (red_i - 1, red_j - 2),
                                                        (red_i, red_j + 1), (red_i - 1, red_j + 1), (red_i - 1, red_j + 2)]
                                    blue_possible_blocking[red_piece] = defensive_moves
                                    red_potential_blocking[red_piece] = defensive_moves

                            else:
                                misalignment = min(abs(red_i - i_blue_1), abs(red_i - i_blue_2))
                                if misalignment < 3:
                                    a = 1 if red_i < min(i_blue_1, i_blue_2) else -1
                                    defensive_moves = [(red_i + a*misalignment + a, red_j), (red_i + a*misalignment + 2*a, red_j),
                                                    (red_i + a*misalignment + a, red_j - a), (red_i + a*misalignment + 2*a, red_j - a),
                                                    (red_i + a*misalignment + a, red_j - 2*a), (red_i + a*misalignment + 2*a, red_j - 2*a)]
                                    blue_possible_blocking[red_piece] = defensive_moves
                                    red_potential_blocking[red_piece] = defensive_moves

                return red_possible_blocking, red_potential_blocking, blue_possible_blocking, blue_potential_blocking


            # region - Check blocking move at edges
            # def check_space_to_edges(last_played_piece, edge):

            #     i,j = last_played_piece
            #     adj_cell = {}
            #     one_cell = {}
            #     both_cell = {}

            #     if edge == 'BOTTOM' or edge == 'RIGHT':
            #         a = 1
            #     elif edge == 'TOP' or edge == 'LEFT':
            #         a = -1
                
            #     if optimal_color == 'R':
            #         solo_adj = [(i-a, j+a), (i, j+a)]
            #         one_gap = [(i-2*a,j+a),(i-2*a,j+2*a), (i-a,j+2*a), (i,j+2*a), (i+1*a,j+a)] #LEFT TO RIGHT

            #         for idx, piece in enumerate(solo_adj):
            #         # If a player MIN has an adjacent piece
            #             if piece in MIN_pieces:
            #                 if one_gap[2] in MIN_pieces:
            #                     if idx == 0:
            #                         both_cell[(last_played_piece, piece, one_gap[2])] = (i+a,j+a)
            #                     else:
            #                         both_cell[(last_played_piece, piece, one_gap[2])] = (i-2*a, j+a)
            #                 else:
            #                     if idx == 0:
            #                         adj_cell[(last_played_piece, piece)] = (solo_adj[1],(i+a, j+a)) # Store the two-bridge pieces and the two solutions
            #                     else:
            #                         adj_cell[(last_played_piece, piece)] = (solo_adj[0],(i-2*a, j+a))
                    
            #         for idx, piece in one_gap:
            #             if piece in MIN_pieces and (last_played_piece, piece) not in both_cell:
            #                 if idx == 2:
            #                     one_cell[(last_played_piece, piece)] = ((i-2*a,j+a), (i+a, j+a))
            #                 elif idx == 0 or idx == 1:
            #                     one_cell[(last_played_piece, piece)] = ((i-a,j+2*a), one_gap[4])
            #                 elif idx == 4 or idx == 3:
            #                     one_cell[(last_played_piece, piece)] = ((i-a,j+2*a), one_gap[0])
            #     else:
            #         solo_adj = [(i+a, j-a), (i+a, j)]
            #         one_gap = [(i+a,j-2*a),(i+2*a,j-2*a), (i+2*a,j-a), (i+2*a,j), (i+a,j+a)] #LEFT TO RIGHT

            #         for idx, piece in enumerate(solo_adj):
            #         # If a player MIN has an adjacent piece
            #             if piece in MIN_pieces:
            #                 if one_gap[2] in MIN_pieces:
            #                     if idx == 0:
            #                         both_cell[(last_played_piece, piece, one_gap[2])] = (i+a,j+a)
            #                     else:
            #                         both_cell[(last_played_piece, piece, one_gap[2])] = (i+a, j-2*a)
            #                 else: # If you have a both configuration then you do not consider the adjacent
            #                     if idx == 0:
            #                         adj_cell[(last_played_piece, piece)] = (solo_adj[1],(i+a, j+a)) # Store the two-bridge pieces and the two solutions
            #                     else:
            #                         adj_cell[(last_played_piece, piece)] = (solo_adj[0],(i+a, j-2*a))
                    
            #         for idx, piece in one_gap:
            #             if piece in MIN_pieces and (last_played_piece, piece) not in both_cell:
            #                 if idx == 2:
            #                     one_cell[(last_played_piece, piece)] = ((i+a,j-2*a), (i+a, j+a))
            #                 elif idx == 0 or idx == 1:
            #                     one_cell[(last_played_piece, piece)] = (one_gap[2], one_gap[4])
            #                 elif idx == 4 or idx == 3:
            #                     one_cell[(last_played_piece, piece)] = (one_gap[2], one_gap[0])

            #     return adj_cell, one_cell, both_cell
            # endregion
        
            # Collect the weakest points of the optimal path
            optimal_path = []
            optimal_color = ''
            adj_cell = {}
            one_cell = {}
            both_cell = {}
            adj_cell2 = {}
            one_cell2 = {}
            both_cell2 = {}

            if MAX_color == 'R':
                MAX_optimal_path = pieces_vertical_path
                MIN_optimal_path = pieces_horizontal_path
                optimal_color = 'R'
            else:
                MAX_optimal_path = pieces_horizontal_path
                MIN_optimal_path = pieces_vertical_path
                optimal_color = 'B'
            
            # region Weak points between a piece and an edge
            # for idx, cell in enumerate(optimal_path):
            #         cell_pos, cell_type = cell
            #         if cell_type == optimal_color:
            #             if idx > 2:
            #                 if optimal_color == 'R':
            #                     adj_cell, one_cell, both_cell = check_space_to_edges(cell_pos, 'BOTTOM')
            #                 else:
            #                     adj_cell, one_cell, both_cell = check_space_to_edges(cell_pos, 'RIGHT')
            #             break
            # for idx, cell in enumerate(reversed(optimal_path)):
            #     cell_pos, cell_type = cell
            #     if cell_type == optimal_color:
            #         if idx > 2:
            #             if optimal_color == 'R':
            #                 adj_cell2, one_cell2, both_cell2 = check_space_to_edges(cell_pos, 'TOP')
            #             else:
            #                 adj_cell2, one_cell2, both_cell2 = check_space_to_edges(cell_pos, 'LEFT')
            #         break
            # adj_cell |= adj_cell2
            # one_cell |= one_cell2
            # both_cell |= both_cell2
            # endregion

            # Weak points between two pieces
            RED_inter_possible_blocking, RED_inter_create_blocking, BLUE_inter_possible_blocking, BLUE_inter_create_blocking = possible_blocking(pieces_vertical_path, pieces_horizontal_path)
            MAX_inter_possible_blocking = {}
            MIN_inter_possible_blocking = {}
            if MAX_color == 'R':
                MAX_inter_possible_blocking = RED_inter_possible_blocking
                MIN_inter_possible_blocking = BLUE_inter_possible_blocking
                MAX_inter_create_blocking = RED_inter_create_blocking
                MIN_inter_create_blocking = BLUE_inter_create_blocking
            else:
                MAX_inter_possible_blocking = BLUE_inter_possible_blocking 
                MIN_inter_possible_blocking = RED_inter_possible_blocking 
                MAX_inter_create_blocking = BLUE_inter_create_blocking
                MIN_inter_create_blocking = RED_inter_create_blocking

            # Partially intercepted two-bridges
            MAX_partially_intercepted_two_bridges, MIN_partially_intercepted_two_bridges, MIN_potential_partially_intercepted_two_bridges = partially_intercepted_two_bridges()

            # Free two bridges
            MAX_free_two_bridges, MAX_potential_free_two_bridges, MIN_free_two_bridges = two_bridge_detection()

            # Weak points definition
            MAX_weak_points = [MAX_partially_intercepted_two_bridges, MAX_inter_possible_blocking]
            MIN_weak_points = [MIN_partially_intercepted_two_bridges, MIN_potential_partially_intercepted_two_bridges, MIN_inter_create_blocking, MIN_inter_possible_blocking]

            # Strong points definition
            MAX_strong_points = [MAX_free_two_bridges, MAX_potential_free_two_bridges]
            
            for i in optimal_path:
                print(i)
            
            return MAX_strong_points, MAX_weak_points, MAX_optimal_path, MIN_free_two_bridges, MIN_weak_points, MIN_optimal_path

        def state_analysis(current_state:GameStateHex):
            board = current_state.get_rep().get_env()
            vertical_optimal_previous_path, horizontal_optimal_previous_path, MAX_pieces, MIN_pieces = shortest_path_computation(current_state)
            MAX_strong_points, MAX_weak_points, MAX_optimal_path, MIN_strong_points, MIN_weak_points, MIN_optimal_path = key_points_identification(current_state, vertical_optimal_previous_path, horizontal_optimal_previous_path, MAX_pieces, MIN_pieces)

            return board, MAX_optimal_path, MIN_optimal_path, MAX_weak_points, MIN_weak_points, MAX_strong_points, MIN_strong_points
        
        # ------------------------------------------------ HEURISTIC ------------------------------------------------

        def heuristic(
            state: GameStateHex,
            MAX_optimal_previous_path: list,
            MIN_optimal_previous_path: list,
            MAX_weak_points: list[dict],
            MIN_weak_points: list[dict],
            MAX_strong_points: list[dict],
            MIN_strong_points: list[dict],
            played_piece: tuple,
            depth: int
        ):

            print('---------OPTIMAL PATH---------')
            for element in MAX_optimal_previous_path:
                print(element)

            # Terminal state
            if state.is_done():
                if state.get_scores()[id_MAX] == 1.0:
                    return 1e6
                else:
                    return -1e6

            # Weak points
            MAX_partially_intercepted_bridges, MAX_potential_inter_blocking = MAX_weak_points
            MIN_partially_intercepted_bridges, MIN_potential_intercepted_two_bridges, MIN_inter_create_blocking, MIN_potential_inter_blocking = MIN_weak_points

            # Strong points
            MAX_confirmed_two_bridges, MAX_potential_two_bridges = MAX_strong_points
            MIN_confirmed_two_bridges = MIN_strong_points

            # Weights
            w_potential_inter_blocking = 12
            w_partially_intercepted_two_bridge = 16
            w_two_bridge = 22

            bonus_attack_opponent_blocking = 18
            bonus_attack_opponent_bridge = 22
            bonus_on_optimal_path = 2
            bonus_on_potential_bridge = 8
            bonus_create_future_interception = 5

            # Counts
            n_MAX_confirmed_two_bridges = len(MAX_confirmed_two_bridges)
            n_MIN_confirmed_two_bridges = len(MIN_confirmed_two_bridges)
            n_MAX_potential_inter_blocking = len(MAX_potential_inter_blocking)
            n_MIN_potential_inter_blocking = len(MIN_potential_inter_blocking)
            n_MAX_partially_intercepted_two_bridges = len(MAX_partially_intercepted_bridges)
            n_MIN_partially_intercepted_two_bridges = len(MIN_partially_intercepted_bridges)

            # Base evaluation
            h_MIN_weak = (
                w_potential_inter_blocking * n_MIN_potential_inter_blocking
                + w_partially_intercepted_two_bridge * n_MIN_partially_intercepted_two_bridges
            )
            h_MIN_strong = w_two_bridge * n_MIN_confirmed_two_bridges

            h_MAX_weak = (
                w_potential_inter_blocking * n_MAX_potential_inter_blocking
                + w_partially_intercepted_two_bridge * n_MAX_partially_intercepted_two_bridges
            )
            h_MAX_strong = w_two_bridge * n_MAX_confirmed_two_bridges

            h_value = (h_MIN_weak + h_MAX_strong) - (h_MIN_strong + h_MAX_weak)

            # Played piece impact
            if depth != limit_depth and played_piece is not None:

                # PLAYED PIECE IMPACT EVALUATION ON WEAK POINTS OF PLAYER MAX
                if MAX_potential_inter_blocking:
                    for possible_answers in MAX_potential_inter_blocking.values():
                        for move in possible_answers:
                            if played_piece == move:  # If we defend the possible blocking
                                h_value += w_potential_inter_blocking

                if MAX_partially_intercepted_bridges:
                    for possible_answer in MAX_partially_intercepted_bridges.values():
                        if possible_answer == played_piece:  # If we defend the possible interception of our two-bridge
                            h_value += w_partially_intercepted_two_bridge

                # PLAYED PIECE IMPACT EVALUATION ON WEAK POINTS OF PLAYER MIN
                if MIN_inter_create_blocking:
                    for possible_blocking in MIN_inter_create_blocking.values():
                        for move in possible_blocking:
                            if played_piece == move:
                                h_value += bonus_attack_opponent_blocking

                if MIN_partially_intercepted_bridges:
                    for possible_answer in MIN_partially_intercepted_bridges.values():
                        if possible_answer == played_piece:  # If we totally intercept the two bridge
                            h_value += bonus_attack_opponent_bridge

                # PLAYED PIECE IMPACT EVALUATION ON STRONG POINTS OF PLAYER MAX
                MAX_path_cells = [cell for cell, _ in MAX_optimal_previous_path]
                if played_piece in MAX_path_cells:
                    h_value += bonus_on_optimal_path

                for potential_bridge in MAX_potential_two_bridges.values():
                    for piece in potential_bridge:
                        if piece == played_piece:
                            h_value += bonus_on_potential_bridge

                # PLAYED PIECE IMPACT EVALUATION ON POTENTIAL NEW WEAK POINTS FOR PLAYER MIN
                if MIN_potential_intercepted_two_bridges:
                    for potential_interception in MIN_potential_intercepted_two_bridges.values():
                        for piece in potential_interception:
                            if piece == played_piece:
                                h_value += bonus_create_future_interception

            return h_value

        # region heuristic old version
        # def heuristic(state: GameStateHex):
        #
        #     empty_cells = []
        #     MAX_color = self.piece_type
        #
        #     if not state.is_done():
        #         free_two_bridges, partially_intercepted_two_bridges, adj_cell, one_cell, both_cell, optimal_path = key_points_identification(state, current_v_opt_path, current_h_opt_path, MAX_color)
        #         h_value = 0
        #
        #         print('FREE TWO-BRIDGES : ', free_two_bridges)
        #         print('PARTIALLY INTERCEPTED TWO-BRIDGES : ',partially_intercepted_two_bridges)
        #         print('ADJ CELL : ',adj_cell)
        #         print('ONE GAP CELL : ',one_cell)
        #         print('BOTH CELL : ',both_cell)
        #
        #         # Good moves treatment
        #         weight_to_complete = 5
        #         weight_free_two_bridge = 0
        #         
        #         for cell in optimal_path:
        #             if cell[1] == 'EMPTY':
        #                 empty_cells.append(cell[0])
        #
        #         for k in range(len(current_opt_path) - 1):
        #             if current_opt_path[k][1] == 'EMPTY' and current_opt_path[k+1][1] == 'EMPTY':
        #                 print('BIG GAP DETECTED')
        #                 weight_free_two_bridge = 5
        #                 weight_to_complete = 0
        #                 break
        #     
        #         n_free_two_bridges = len(free_two_bridges)
        #         n_to_complete = len(empty_cells)
        #
        #         # Weak points treatment
        #         weight_partially_int_two_bridge = 50
        #         weight_edge_threat = 100
        #         n_partially_int_two_bridge = len(partially_intercepted_two_bridges)
        #         n_edge_threats = len(adj_cell) + len(one_cell) + len(both_cell)
        #
        #         h_value = n_free_two_bridges*weight_free_two_bridge - n_to_complete*weight_to_complete - 0*n_edge_threats - n_partially_int_two_bridge*weight_partially_int_two_bridge
        #
        #         print(h_value)
        #         return h_value
        #     
        #     else:
        #         if state.get_scores()[id_MAX] == 1.0:
        #             return 1e6
        #         else:
        #             return -1e6
        # endregion
        
        # ------------------------------------------------ MIN/MAX - ALPHA/BETA - ALGORITHM ------------------------------------------------
        # For the advanced moves

        board = current_state.get_rep().get_env()
        current_v_opt_path, current_h_opt_path, current_MAX, current_MIN = shortest_path_computation(current_state)
        MAX_color = self.piece_type

        if MAX_color == 'R':
            current_opt_path = current_v_opt_path
            MIN_color = 'B'
        else:
            current_opt_path = current_h_opt_path
            MIN_color = 'R'

        depth = current_state.get_step()
        limit_depth = depth + 6
        width = 5
        id_MAX = current_state.active_player.get_id()
        first_moves = list(range(4))

        if depth == 1:
            first_MIN_piece = next(iter(board))

        first_MAX_piece = ()
        if depth == 3:
            for piece, piece_type in board.items():
                if piece_type.get_owner_id() == id_MAX:
                    first_MAX_piece = piece
        
        if depth not in first_moves:

            def player_MAX(current_state, alpha, beta, depth):
                
                if current_state.is_done() and current_state.get_scores()[id_MAX] == 1.0:
                    return (10e6, None)
                elif current_state.is_done() and current_state.get_scores()[id_MAX] == 0.0:
                    return(-10e6, None)
                elif depth == limit_depth:
                    board, MAX_optimal_path, MIN_optimal_path, MAX_weak_points, MIN_weak_points, MAX_strong_points, MIN_strong_points = state_analysis(current_state)
                    return (heuristic(current_state,MAX_optimal_path, MIN_optimal_path, MAX_weak_points, MIN_weak_points, MAX_strong_points, MIN_strong_points, None, depth), None)
                else:
                    board, MAX_optimal_path, MIN_optimal_path, MAX_weak_points, MIN_weak_points, MAX_strong_points, MIN_strong_points = state_analysis(current_state)
                    best_estimation = -np.inf
                    best_action = None
                    all_actions = list(current_state.generate_possible_stateful_actions())
                    number_of_actions = len(all_actions)
                    if number_of_actions >= width:
                        selected_actions = [(None, -np.inf) for _ in range(width)]
                    else: 
                        selected_actions = [(None, -np.inf) for _ in range(number_of_actions)]

                    for action in all_actions:
                        idx, value = min(enumerate(selected_actions), key=lambda x: x[1][1])
                        next_board = action.next_game_state.get_rep().get_env()
                        played_piece = next(piece for piece in next_board if piece not in board)
                        h_value = heuristic(action.next_game_state, MAX_optimal_path, MIN_optimal_path, MAX_weak_points, MIN_weak_points, MAX_strong_points, MIN_strong_points, played_piece, depth)
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
                    return (10e6, None)
                elif current_state.is_done() and current_state.get_scores()[id_MAX] == 0.0:
                    return(-10e6, None)
                elif depth == limit_depth:
                    board, MAX_optimal_path, MIN_optimal_path, MAX_weak_points, MIN_weak_points, MAX_strong_points, MIN_strong_points = state_analysis(current_state)
                    return (heuristic(current_state,MAX_optimal_path, MIN_optimal_path, MAX_weak_points, MIN_weak_points, MAX_strong_points, MIN_strong_points, None, depth), None)
                else:
                    board, MAX_optimal_path, MIN_optimal_path, MAX_weak_points, MIN_weak_points, MAX_strong_points, MIN_strong_points = state_analysis(current_state)

                    best_estimation = np.inf
                    best_action = None
                    all_actions = list(current_state.generate_possible_stateful_actions())
                    number_of_actions = len(all_actions)
                    if number_of_actions >= width:
                        selected_actions = [(None, np.inf) for _ in range(width)]
                    else:
                        selected_actions = [(None, np.inf) for _ in range(number_of_actions)]
                    
                    for action in all_actions:
                        idx, value = max(enumerate(selected_actions), key=lambda x: x[1][1])
                        next_board = action.next_game_state.get_rep().get_env()
                        played_piece = next(piece for piece in next_board if piece not in board)
                        h_value = heuristic(action.next_game_state, MAX_optimal_path, MIN_optimal_path, MAX_weak_points, MIN_weak_points, MAX_strong_points, MIN_strong_points, played_piece, depth)
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
                    if (6,6) in action.get_next_game_state().get_rep().get_env():
                        return action
                    
            elif depth == 1: # => player MAX is a BLUE player - first move
                upper_part = range(7)
                lower_part = range(7,14)
                i,j = first_MIN_piece
                if i in upper_part:
                    if i == 0:
                        for action in current_state.generate_possible_stateful_actions():
                                    if (i+2,j+1) in action.get_next_game_state().get_rep().get_env():
                                        return action
                    else:
                        for action in current_state.generate_possible_stateful_actions():
                                    if (i+2,j) in action.get_next_game_state().get_rep().get_env():
                                        return action
                    
                elif i in lower_part:
                    for action in current_state.generate_possible_stateful_actions():
                                if (i-2,j) in action.get_next_game_state().get_rep().get_env():
                                    return action

            elif depth == 2: # => player MAX is a RED player - second move
                MIN_piece = next(iter(current_MIN))
                upper_part = range(7)
                i,j = MIN_piece
                if MIN_piece == (5,6):
                    for action in current_state.generate_possible_stateful_actions():
                                if (5,8) in action.get_next_game_state().get_rep().get_env():
                                    return action
                elif MIN_piece == (5,7) or MIN_piece == (4,7):
                    for action in current_state.generate_possible_stateful_actions():
                                if (5,5) in action.get_next_game_state().get_rep().get_env():
                                    return action
                elif MIN_piece == (7,5) or MIN_piece == (8,5):
                    for action in current_state.generate_possible_stateful_actions():
                                if (7,7) in action.get_next_game_state().get_rep().get_env():
                                    return action
                elif MIN_piece == (7,6):
                    for action in current_state.generate_possible_stateful_actions():
                                if (7,4) in action.get_next_game_state().get_rep().get_env():
                                    return action
                else:
                    if i in upper_part:
                        for action in current_state.generate_possible_stateful_actions():
                                    if (4,7) in action.get_next_game_state().get_rep().get_env():
                                        return action
                    else:
                        for action in current_state.generate_possible_stateful_actions():
                                    if (8,5) in action.get_next_game_state().get_rep().get_env():
                                        return action
            
            elif depth == 3:
                i,j = first_MAX_piece
                for action in current_state.generate_possible_stateful_actions():
                                    if (i-1,j+2) in action.get_next_game_state().get_rep().get_env():
                                        return action
                                    elif (i+1,j+1) in action.get_next_game_state().get_rep().get_env():
                                        return action

        raise MethodNotImplementedError()
