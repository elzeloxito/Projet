from player_hex import PlayerHex
from seahorse.game.action import Action
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import numpy as np
import random
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
        # Gives the different groups 
        def group_detection(state:GameStateHex, RED_pieces, BLUE_pieces):

            RED_groups = [] # list of sets that contain the RED pieces of a given group
            visited_states = set()

            for piece in RED_pieces:
                if piece not in visited_states:
                    fringe = set()
                    fringe.add(piece)
                    group = set()
                    while fringe: 
                        current_piece = fringe.pop()
                        if current_piece not in visited_states:
                            group.add(current_piece)
                            visited_states.add(current_piece)
                            i,j = current_piece
                            for neighbor_type, neighbor_pos in state.get_neighbours(i,j).values(): 
                                if neighbor_type == 'R' and neighbor_pos not in visited_states:  
                                    fringe.add(neighbor_pos)
                    RED_groups.append(group)

            BLUE_groups = [] # list of sets that contain the RED pieces of a given group
            visited_states = set()

            for piece in BLUE_pieces:
                if piece not in visited_states:
                    fringe = set()
                    fringe.add(piece)
                    group = set()
                    while fringe: 
                        current_piece = fringe.pop()
                        if current_piece not in visited_states:
                            group.add(current_piece)
                            visited_states.add(current_piece)
                            i,j = current_piece
                            for neighbor_type, neighbor_pos in state.get_neighbours(i,j).values(): 
                                if neighbor_type == 'B' and neighbor_pos not in visited_states:  
                                    fringe.add(neighbor_pos)
                    BLUE_groups.append(group)

            return RED_groups, BLUE_groups
        
        # Gives the orientation of a given group considering its color
        def group_orientation(group, group_color):
            # For a RED group
            if group_color == 'R':
                i_sup = -np.inf
                i_inf = np.inf
                for piece in group:
                    i,j = piece
                    if i > i_sup:
                        i_sup = i
                    if i < i_inf:
                        i_inf = i
                inf = i_inf
                sup = i_sup
            
            # For a BLUE group
            else:
                j_sup = -np.inf
                j_inf = np.inf
                for piece in group:
                    i,j = piece
                    if j > j_sup:
                        j_sup = j
                    if j < j_inf:
                        j_inf = j
                inf = j_inf 
                sup = j_sup

            return inf, sup
        
        # Detection of different types of two_bridges 
        # The weak point is only the intercepted two-bridge 
        def two_bridge_detection(side, central_cell, RED_pieces, BLUE_pieces, group_color, board):
            i,j = central_cell
            line_out = [-1, -2, -1, 1, 2, 1]
            column_out = [-1, 1, 2, 1, -1, -2]
            line_in = [0, -1, -1, 0, 1, 1]
            column_in = [-1, 0, 1, 1, 0, -1]

            outer_crown = [(i + dx, j + dy) for dx, dy in zip(line_out, column_out)]
            inner_crown = [(i + dx, j + dy) for dx, dy in zip(line_in, column_in)]
            free_two_bridge = {}
            partially_intercepted_two_bridge = {}
            completed_two_bridges = {}
            
            if group_color == 'R':
                if side == 'TOP':
                    for idx in [0,1,2]:
                        cell = outer_crown[idx]
                        # Two-bridges
                        if cell in RED_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] not in board:
                            free_two_bridge[((central_cell, cell))] = True
                        # Partially intercepted two-bridges and completed ones
                        elif cell in RED_pieces and inner_crown[idx] in BLUE_pieces and inner_crown[(idx + 1) % 6] not in board:
                            partially_intercepted_two_bridge[(central_cell, cell)] = inner_crown[(idx + 1) % 6]
                        elif cell in RED_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] in BLUE_pieces:
                            partially_intercepted_two_bridge[(central_cell, cell)] = inner_crown[idx]
                        elif cell in RED_pieces and (inner_crown[idx] in RED_pieces or inner_crown[(idx + 1) % 6] in RED_pieces) and outer_crown[idx] not in RED_pieces and outer_crown[(idx + 1) % 6]:
                            completed_two_bridges[(central_cell, cell)] = True
                else: # side == 'BOTTOM'
                    for idx in [3,4,5]:
                        cell = outer_crown[idx]
                        # Two-bridges
                        if cell in RED_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] not in board:
                            free_two_bridge[((central_cell, cell))] = True
                        # Partially intercepted two-bridges and completed ones
                        elif cell in RED_pieces and inner_crown[idx] in BLUE_pieces and inner_crown[(idx + 1) % 6] not in board:
                            partially_intercepted_two_bridge[(central_cell, cell)] = inner_crown[(idx + 1) % 6]
                        elif cell in RED_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] in BLUE_pieces:
                            partially_intercepted_two_bridge[(central_cell, cell)] = inner_crown[idx]
                        elif cell in RED_pieces and (inner_crown[idx] in RED_pieces or inner_crown[(idx + 1) % 6] in RED_pieces) and outer_crown[idx] not in RED_pieces and outer_crown[(idx + 1) % 6]:
                            completed_two_bridges[(central_cell, cell)] = True

            else:
                if side == 'LEFT':
                    for idx in [0, 5]:
                        cell = outer_crown[idx]
                        if cell in BLUE_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] not in board:
                            free_two_bridge[(central_cell, cell)] = True
                        elif cell in BLUE_pieces and inner_crown[idx] in RED_pieces and inner_crown[(idx + 1) % 6] not in board:
                            partially_intercepted_two_bridge[(central_cell, cell)] = inner_crown[(idx + 1) % 6]
                        elif cell in BLUE_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] in RED_pieces:
                            partially_intercepted_two_bridge[(central_cell, cell)] = inner_crown[idx]
                        elif cell in BLUE_pieces and (inner_crown[idx] in BLUE_pieces or inner_crown[(idx + 1) % 6] in BLUE_pieces) and outer_crown[idx] not in BLUE_pieces and outer_crown[(idx + 1) % 6]:
                            completed_two_bridges[(central_cell, cell)] = True
                else: # side == 'RIGHT'
                    for idx in [2, 3]:
                        cell = outer_crown[idx]
                        if cell in BLUE_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] not in board:
                            free_two_bridge[(central_cell, cell)] = True
                        elif cell in BLUE_pieces and inner_crown[idx] in RED_pieces and inner_crown[(idx + 1) % 6] not in board:
                            partially_intercepted_two_bridge[(central_cell, cell)] = inner_crown[(idx + 1) % 6]
                        elif cell in BLUE_pieces and inner_crown[idx] not in board and inner_crown[(idx + 1) % 6] in RED_pieces:
                            partially_intercepted_two_bridge[(central_cell, cell)] = inner_crown[idx]
                        elif cell in BLUE_pieces and (inner_crown[idx] in BLUE_pieces or inner_crown[(idx + 1) % 6] in BLUE_pieces) and outer_crown[idx] not in BLUE_pieces and outer_crown[(idx + 1) % 6]:
                            completed_two_bridges[(central_cell, cell)] = True

            return free_two_bridge, partially_intercepted_two_bridge, completed_two_bridges
        
        def group_connexion(group, group_color, RED_pieces, BLUE_pieces, board):

            two_bridge_coefficient = 0
            part_int_two_bridge_coefficient = 0

            if group_color == 'R':
                sup, inf = group_orientation(group, group_color) # high column number => bottom of the board and vice versa
                group_middle = (sup + inf)/2
                first_half = sorted([x for x in group if x[0] <= group_middle], key=lambda x: x[0])
                second_half = sorted([x for x in group if x[0] >= group_middle], key=lambda x: x[0], reverse=True)
                group_diameter = abs(sup-inf) if sup != inf else 1 
                for piece in first_half:
                    i,j = piece
                    distance_to_top = abs(i-sup)
                    free_two_bridge, partially_intercepted_two_bridge, completed_two_bridges = two_bridge_detection('TOP', piece, RED_pieces, BLUE_pieces, group_color, board)
                    if partially_intercepted_two_bridge and i == sup:
                        part_int_two_bridge_coefficient = 1
                    if free_two_bridge:
                        two_bridge_coefficient += (group_diameter - distance_to_top)/group_diameter
                        break

                for piece in second_half:
                    i,j = piece
                    distance_to_bottom = abs(i-inf)
                    free_two_bridge, partially_intercepted_two_bridge, completed_two_bridges = two_bridge_detection('BOTTOM', piece, RED_pieces, BLUE_pieces, group_color, board)
                    if partially_intercepted_two_bridge and i == inf:
                        part_int_two_bridge_coefficient = 1
                    if free_two_bridge:
                        two_bridge_coefficient += (group_diameter - distance_to_bottom)/group_diameter
                        break
            else:
                inf, sup = group_orientation(group, group_color) #inf => LEFT sup => RIGHT
                group_middle = (sup + inf)/2
                first_half = sorted([x for x in group if x[0] <= group_middle], key=lambda x: x[0]) # Left part 
                second_half = sorted([x for x in group if x[0] >= group_middle], key=lambda x: x[0], reverse=True) # Right part
                group_diameter = abs(sup-inf) if sup != inf else 1
                for piece in first_half:
                    i,j = piece
                    distance_to_top = abs(j-sup)
                    free_two_bridge, partially_intercepted_two_bridge, completed_two_bridges = two_bridge_detection('LEFT', piece, RED_pieces, BLUE_pieces, group_color, board)
                    if partially_intercepted_two_bridge and j == inf:
                        part_int_two_bridge_coefficient = 1
                    if free_two_bridge:
                        two_bridge_coefficient += (group_diameter - distance_to_top)/group_diameter
                        break

                for piece in second_half:
                    i,j = piece
                    distance_to_bottom = abs(j-sup)
                    free_two_bridge, partially_intercepted_two_bridge, completed_two_bridges = two_bridge_detection('RIGHT', piece, RED_pieces, BLUE_pieces, group_color, board)
                    if partially_intercepted_two_bridge and j == sup:
                        part_int_two_bridge_coefficient = 1
                    if free_two_bridge:
                        two_bridge_coefficient += (group_diameter - distance_to_bottom)/group_diameter
                        break

            return two_bridge_coefficient, part_int_two_bridge_coefficient
        
        # Check if a group is blocked or not
        def blocking_detection(RED_pieces, BLUE_pieces, group, group_color):

            blocked_pieces = {}

            # Red pieces are blocked
            if group_color == 'R':
                top_pieces = set()
                bottom_pieces = set()
                sup, inf = group_orientation(group, group_color)
                for piece in group:
                    i,j = piece
                    if i == sup:
                        top_pieces.add(piece)
                    elif i == inf:
                        bottom_pieces.add(piece)
                for piece in top_pieces:
                    i,j = piece
                    # Upper
                    # Classical blocking 1
                    empty_pieces_upper1 = [(i-1,j-1), (i-1,j), (i-1,j+1), (i-1,j+2), (i-2,j), (i-2,j+2)]
                    empty_pieces_upper2 = [(i-1,j+2), (i-1,j+1), (i-2,j+1), (i-1,j-1)]
                    empty_pieces_upper3 = [(i-1,j-1), (i-1,j), (i-2,j+1),(i-1,j+1)]
                    if (i-1,j) in BLUE_pieces and (i-1,j+1) in BLUE_pieces:
                        blocked_pieces[piece] = True
                    elif not any(pos in RED_pieces for pos in empty_pieces_upper1):
                        if (((i,j-1) in BLUE_pieces and (i,j+1) in BLUE_pieces and (i-2,j+1) in BLUE_pieces) and ((i+1,j-1) in RED_pieces or (i+1,j) in RED_pieces)):
                            blocked_pieces[piece] = True
                        elif ((i,j-1) in BLUE_pieces and (i-2,j+1) in BLUE_pieces) or ((i,j+1) in BLUE_pieces and (i-2,j+1) in BLUE_pieces):
                            blocked_pieces[piece] = True
                    elif not any(pos in RED_pieces for pos in empty_pieces_upper2):
                        if (i-1,j) in BLUE_pieces and (i-2,j+2) in BLUE_pieces:
                            blocked_pieces[piece] = True
                    elif not any(pos in RED_pieces for pos in empty_pieces_upper3):
                        if (i-1,j+1) in BLUE_pieces and (i-2,j) in BLUE_pieces:
                            blocked_pieces[piece] = True
                for piece in bottom_pieces:
                    i,j = piece
                    # Lower
                    # Classical blocking 1
                    empty_pieces1 = [(i+1,j-2), (i+1,j-1), (i+1,j), (i+1,j+1), (i+2,j), (i+2,j-2)]
                    empty_pieces2 = [(i+1,j-2), (i+1,j-1), (i+2,j-1), (i+1,j+1)]
                    empty_pieces3 = [(i+1,j+1), (i+1,j), (i+2,j-1), (i+1,j-2)]
                    if (i+1,j-1) in BLUE_pieces and (i+1,j) in BLUE_pieces:
                        blocked_pieces[piece] = True
                    elif not any(pos in RED_pieces for pos in empty_pieces1):
                        # Combined block 2
                        if (((i,j+1) in BLUE_pieces and (i,j-1) in BLUE_pieces and (i+2,j-1) in BLUE_pieces) and ((i-1,j+1) in RED_pieces or (i-1,j) in RED_pieces)):
                            blocked_pieces[piece] = True
                        # Combined block 1
                        elif ((i,j+1) in BLUE_pieces and (i+2,j-1) in BLUE_pieces) or ((i,j-1) in BLUE_pieces and (i+2,j-1) in BLUE_pieces):
                            blocked_pieces[piece] = True
                    elif not any(pos in RED_pieces for pos in empty_pieces2):
                        # Classical block 2
                        if (i+1,j) in BLUE_pieces and (i+2,j-2) in BLUE_pieces:
                            blocked_pieces[piece] = True
                    elif not any(pos in RED_pieces for pos in empty_pieces3):
                        if (i+1,j-1) in BLUE_pieces and (i+2,j) in BLUE_pieces:
                            blocked_pieces[piece] = True
            
            else:
                left_pieces = set()
                right_pieces = set()
                left, right = group_orientation(group, group_color)
                for piece in group:
                    i,j = piece
                    if j == left:
                        left_pieces.add(piece)
                    elif j == right:
                        right_pieces.add(piece)

                for piece in left_pieces:
                    i,j = piece
                    # Left
                    empty_pieces_left1 = [(i-1,j-1), (i,j-1), (i+1,j-1), (i+2,j-1), (i,j-2), (i+2,j-2)]
                    empty_pieces_left2 = [(i+2,j-1), (i+1,j-1), (i+1,j-2), (i-1,j-1)]
                    empty_pieces_left3 = [(i-1,j-1), (i,j-1), (i+1,j-2), (i+1,j-1)]
                    if (i,j-1) in RED_pieces and (i+1,j-1) in RED_pieces:
                        blocked_pieces[piece] = True
                    elif not any(pos in BLUE_pieces for pos in empty_pieces_left1):
                        if (((i-1,j) in RED_pieces and (i+1,j) in RED_pieces and (i+1,j-2) in RED_pieces) and ((i-1,j+1) in BLUE_pieces or (i,j+1) in BLUE_pieces)):
                            blocked_pieces[piece] = True
                        elif ((i-1,j) in RED_pieces and (i+1,j-2) in RED_pieces) or ((i+1,j) in RED_pieces and (i+1,j-2) in RED_pieces):
                            blocked_pieces[piece] = True
                    elif not any(pos in BLUE_pieces for pos in empty_pieces_left2):
                        if (i,j-1) in RED_pieces and (i+2,j-2) in RED_pieces:
                            blocked_pieces[piece] = True
                    elif not any(pos in BLUE_pieces for pos in empty_pieces_left3):
                        if (i+1,j-1) in RED_pieces and (i,j-2) in RED_pieces:
                            blocked_pieces[piece] = True

                for piece in right_pieces:
                    i,j = piece
                    empty_pieces_right1 = [(i-1,j+1), (i,j+1), (i+1,j+1), (i+2,j+1), (i,j+2), (i+2,j+2)]
                    empty_pieces_right2 = [(i+2,j+1), (i+1,j+1), (i+1,j+2), (i-1,j+1)]
                    empty_pieces_right3 = [(i-1,j+1), (i,j+1), (i+1,j+2), (i+1,j+1)]
                    if (i,j+1) in RED_pieces and (i+1,j+1) in RED_pieces:
                        blocked_pieces[piece] = True
                    elif not any(pos in BLUE_pieces for pos in empty_pieces_right1):
                        if (((i-1,j) in RED_pieces and (i+1,j) in RED_pieces and (i+1,j+2) in RED_pieces) and ((i-1,j-1) in BLUE_pieces or (i,j-1) in BLUE_pieces)):
                            blocked_pieces[piece] = True
                        elif ((i-1,j) in RED_pieces and (i+1,j+2) in RED_pieces) or ((i+1,j) in RED_pieces and (i+1,j+2) in RED_pieces):
                            blocked_pieces[piece] = True
                    elif not any(pos in BLUE_pieces for pos in empty_pieces_right2):
                        if (i,j+1) in RED_pieces and (i+2,j+2) in RED_pieces:
                            blocked_pieces[piece] = True
                    elif not any(pos in BLUE_pieces for pos in empty_pieces_right3):
                        if (i+1,j+1) in RED_pieces and (i,j+2) in RED_pieces:
                            blocked_pieces[piece] = True

            return blocked_pieces

        # EDGE DETECTION 
        def edge_connexion(RED_pieces, BLUE_pieces, group, group_color, board):

            free_edge = set()
            int_edge = set()
            confirmed_edge = set()

            if group_color == 'R':
                top_pieces = set()
                bottom_pieces = set()
                sup, inf = group_orientation(group, group_color)
                for piece in group:
                    i,j = piece
                    if i == sup:
                        top_pieces.add(piece)
                    elif i == inf:
                        bottom_pieces.add(piece)

                for piece in top_pieces:
                    i,j = piece 
                    if i == 1 and (0,j) not in board and (0,j+1) not in board and (1,j-1) not in RED_pieces and (1,j+1) not in RED_pieces:
                        free_edge.add(piece)
                    elif i == 1 and (1,j-1) in BLUE_pieces and (0,j+1) not in board:
                        int_edge.add(piece)
                    elif i == 1 and (0,j) in RED_pieces and (0,j+1) not in RED_pieces and (1,j-1) not in RED_pieces:
                        confirmed_edge.add(piece)
                    elif i == 1 and (0,j+1) in RED_pieces and (0,j) not in RED_pieces and (1,j+1) not in RED_pieces:
                        confirmed_edge.add(piece)

                for piece in bottom_pieces:
                    i,j = piece
                    if i == 12 and (13,j-1) not in board and (13,j) not in board and (12,j-1) not in RED_pieces and (12,j+1) not in RED_pieces:
                        free_edge.add(piece)
                    elif i == 12 and (12,j+1) in BLUE_pieces and (13,j-1) not in board:
                        int_edge.add(piece)
                    elif i == 12 and (13,j-1) in RED_pieces and (13,j) not in RED_pieces and (12,j-1) not in RED_pieces:
                        confirmed_edge.add(piece)
                    elif i == 12 and (13,j) in RED_pieces and (13,j-1) not in RED_pieces and (12,j+1) not in RED_pieces:
                        confirmed_edge.add(piece)
                
            else:
                left_pieces = set()
                right_pieces = set()
                left, right = group_orientation(group, group_color)
                for piece in group:
                    i,j = piece
                    if j == left:
                        left_pieces.add(piece)
                    elif j == right:
                        right_pieces.add(piece)
                
                for piece in left_pieces:
                    i,j = piece
                    if j == 1 and (i,0) not in board and (i+1,0) not in board and (i-1,1) not in BLUE_pieces and (i+1,1) not in BLUE_pieces:
                        free_edge.add(piece)
                    elif j == 1 and (i-1,1) in RED_pieces and (i+1,0) not in board:
                        int_edge.add(piece)
                    elif j == 1 and (i,0) in BLUE_pieces and (i+1,0) not in BLUE_pieces and (i-1,1) not in BLUE_pieces:
                        confirmed_edge.add(piece)
                    elif j == 1 and (i+1,0) in BLUE_pieces and (i,0) not in BLUE_pieces and (i+1,1) not in BLUE_pieces:
                        confirmed_edge.add(piece)

                for piece in right_pieces:
                    i,j = piece
                    if j == 12 and (i,13) not in board and (i-1,13) not in board and (i-1,12) not in BLUE_pieces and (i+1,12) not in BLUE_pieces:
                        free_edge.add(piece)
                    elif j == 12 and (i+1,12) in RED_pieces and (i-1,13) not in board:
                        int_edge.add(piece)
                    elif j == 12 and (i,13) in BLUE_pieces and (i-1,13) not in BLUE_pieces and (i+1,12) not in BLUE_pieces:
                        confirmed_edge.add(piece)
                    elif j == 12 and (i-1,13) in BLUE_pieces and (i,13) not in BLUE_pieces and (i-1,12) not in BLUE_pieces:
                        confirmed_edge.add(piece)

            return free_edge, int_edge, confirmed_edge
    
        # ------------------------------------------------ HEURISTIC ------------------------------------------------
        def heuristic(state: GameStateHex):

            MAX_color = self.piece_type
            board = state.get_rep().get_env()
            id_MAX = current_state.active_player.get_id()
            MAX_pieces = set()
            MIN_pieces = set()

            for piece, piece_type in board.items():
                if piece_type.get_owner_id() == id_MAX:
                    MAX_pieces.add(piece)
                else:
                    MIN_pieces.add(piece)
            
            if MAX_color == 'R':
                RED_pieces = MAX_pieces
                BLUE_pieces = MIN_pieces
            else:
                RED_pieces = MIN_pieces
                BLUE_pieces = MAX_pieces

            h_red = 0
            h_blue = 0
            h_value = 0

            w_free_two_bridge = 3
            w_part_int_two_bridge = 5
            w_free_edge = 1
            w_int_edge = 5
            w_confirmed_edge = 3
            w_blocked_pieces = 8

            if not state.is_done():
                # Groups detection 
                RED_groups, BLUE_groups = group_detection(state, RED_pieces, BLUE_pieces)

                # Groups characteristics
                for group in RED_groups:
                    group_color = 'R'
                    # Group orientation
                    top, bottom = group_orientation(group, group_color)
                    orientation = abs(top - bottom)

                    # Group connexion to other groups
                    two_bridge_coefficient, partially_intercepted_coefficient = group_connexion(group, group_color, RED_pieces, BLUE_pieces, board)

                    # Group connexion to edge
                    free_edge, int_edge, confirmed_edge = edge_connexion(RED_pieces, BLUE_pieces, group, group_color, board)

                    # Group blocking
                    blocked_pieces = blocking_detection(RED_pieces, BLUE_pieces, group, group_color)

                    h_red += orientation + w_free_two_bridge*two_bridge_coefficient - w_part_int_two_bridge*partially_intercepted_coefficient + w_free_edge*len(free_edge) - w_int_edge*len(int_edge) + w_confirmed_edge*len(confirmed_edge) - w_blocked_pieces*len(blocked_pieces)

                for group in BLUE_groups:
                    group_color = 'B'
                    # Group orientation
                    left, right = group_orientation(group, group_color)
                    orientation = abs(left - right)

                    # Group connexion to other groups
                    two_bridge_coefficient, partially_intercepted_coefficient = group_connexion(group, group_color, RED_pieces, BLUE_pieces, board)

                    # Group connexion to edge
                    free_edge, int_edge, confirmed_edge = edge_connexion(RED_pieces, BLUE_pieces, group, group_color, board)

                    # Group blocking
                    blocked_pieces = blocking_detection(RED_pieces, BLUE_pieces, group, group_color)

                    h_blue += orientation + w_free_two_bridge*two_bridge_coefficient - w_part_int_two_bridge*partially_intercepted_coefficient + w_free_edge*len(free_edge) - w_int_edge*len(int_edge) + w_confirmed_edge*len(confirmed_edge) - w_blocked_pieces*len(blocked_pieces)

                if MAX_color == 'R':
                    h_value = h_red - h_blue
                else:
                    h_value = h_blue - h_red
            
            else:
                if state.get_scores()[id_MAX] == 1.0:
                    h_value = 1e6
                else:
                    h_value = -1e6
            
            print(h_value)
            return h_value
        
        # ------------------------------------------------ MIN/MAX - ALPHA/BETA - ALGORITHM ------------------------------------------------
        # For the advanced moves

        board = current_state.get_rep().get_env()

        depth = current_state.get_step()
        if depth <= 14:
            limit_depth = depth + 5
            width = 5
        else: 
            limit_depth = depth + 5
            width = 5
        id_MAX = current_state.active_player.get_id()
        first_moves = list(range(4))

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
            # ------------------------------------------------ OPENING MOVES ------------------------------------------------
            # For the first moves 
            MAX_color = self.piece_type #couleur du joueur MAX
            
            if depth == 0: # => player MAX is a RED player - first move
                for action in current_state.generate_possible_stateful_actions():
                    if (10,3) in action.get_next_game_state().get_rep().get_env():
                        return action #11d
                    
            elif depth == 1: # => player MAX is a BLUE player - first move
                MIN_piece = next(iter(board))
                i,j = MIN_piece
                if MIN_piece == (10,3): #11d
                    for action in current_state.generate_possible_stateful_actions():
                        if (3,10) in action.get_next_game_state().get_rep().get_env():
                            return action #4k 
                elif MIN_piece == (3,10): #sym 4k
                    for action in current_state.generate_possible_stateful_actions():
                        if (10,3) in action.get_next_game_state().get_rep().get_env():
                            return action #11d
                elif i+j == 13: #diag
                    if i<j:
                        for action in current_state.generate_possible_stateful_actions():
                            if (10,3) in action.get_next_game_state().get_rep().get_env():
                                return action #11d
                    else:
                        for action in current_state.generate_possible_stateful_actions():
                            if (3,10) in action.get_next_game_state().get_rep().get_env():
                                return action #4k
                elif i+j == 12:
                    for action in current_state.generate_possible_stateful_actions():
                        if (i+1,j) in action.get_next_game_state().get_rep().get_env():
                            return action #checked
                elif i+j == 14:
                    for action in current_state.generate_possible_stateful_actions():
                        if (i-1,j) in action.get_next_game_state().get_rep().get_env():
                            return action #checked
                elif i+j<= 11 and i+j !=13:
                    for action in current_state.generate_possible_stateful_actions():
                        if (i+1,j) in action.get_next_game_state().get_rep().get_env():
                            return action
                elif j+i >= 15 and i+j !=13:
                    for action in current_state.generate_possible_stateful_actions():
                        if (i-1,j) in action.get_next_game_state().get_rep().get_env():
                            return action

            elif depth == 2: # => player MAX is a RED player - second move
                # first move d11
                MIN_piece = next(iter(board))
                #i,j = MIN_piece
                if MIN_piece == (3,10): #4k
                    for action in current_state.generate_possible_stateful_actions():
                        if (3,9) in action.get_next_game_state().get_rep().get_env():
                            return action #4j
                elif MIN_piece == (10,3): #11d sym
                    for action in current_state.generate_possible_stateful_actions():
                        if (10,4) in action.get_next_game_state().get_rep().get_env():
                            return action #10d
                elif MIN_piece == (2,10): #3k
                    for action in current_state.generate_possible_stateful_actions():
                        if (4,3) in action.get_next_game_state().get_rep().get_env():
                            return action #5d
                elif MIN_piece == (10,2): #sym 11c
                    for action in current_state.generate_possible_stateful_actions():
                        if (9,10) in action.get_next_game_state().get_rep().get_env():
                            return action #10k
                else:
                    for action in current_state.generate_possible_stateful_actions():
                        if (3,10) in action.get_next_game_state().get_rep().get_env():
                            return action #4k
            
            elif depth == 3: #Cette partie n'est vraiment pas folle, je pense qu'un min max peut �tre mieux. Il y a tellement de possibilit�s, j'ai mis un peu d'al�atoire, � voir ce que ca donne en pratique
                board = current_state.get_rep().get_env()
                red_pieces = []
                blue_pieces = []

                for pos, piece in board.items():
                    if piece.get_type() == 'R':
                        red_pieces.append(pos) #position MIN
                    elif piece.get_type() == 'B':
                        blue_pieces.append(pos)#position MAX
                
                if (red_pieces[0] == (10,3) and blue_pieces[0]==(3,10) and red_pieces[1]==(6,7)) or (red_pieces[1] == (10,3) and blue_pieces[0]==(3,10) and red_pieces[0]==(6,7)):
                    for action in current_state.generate_possible_stateful_actions():
                        if (11,2) in action.get_next_game_state().get_rep().get_env():
                            return action
                elif (red_pieces[0] == (3,10) and blue_pieces[0]==(10,3) and red_pieces[1]==(7,6)) or (red_pieces[1] == (3,10) and blue_pieces[0]==(3,10) and red_pieces[0]==(7,6)):
                    for action in current_state.generate_possible_stateful_actions():
                        if (2,11) in action.get_next_game_state().get_rep().get_env():
                            return action
                elif (red_pieces[0] == (10,3) and blue_pieces[0]==(3,10) and red_pieces[1]==(4,3)) or (red_pieces[1] == (10,3) and blue_pieces[0]==(3,10) and red_pieces[0]==(4,3)):
                    for action in current_state.generate_possible_stateful_actions():
                        if (7,3) in action.get_next_game_state().get_rep().get_env():
                            return action
                elif (red_pieces[0] == (3,10) and blue_pieces[0]==(10,3) and red_pieces[1]==(3,4)) or (red_pieces[1] == (3,10) and blue_pieces[0]==(10,3) and red_pieces[0]==(3,4)):
                    for action in current_state.generate_possible_stateful_actions():
                        if (3,7) in action.get_next_game_state().get_rep().get_env():
                            return action
                elif (abs(red_pieces[0][1] - red_pieces[1][1]) == 1) and (red_pieces[0][0]==red_pieces[1][0]):
                    if red_pieces[0][0] < red_pieces[1][0]:
                        if blue_pieces[0][0] == red_pieces[0][0]:
                            for action in current_state.generate_possible_stateful_actions():
                                if (blue_pieces[0][0], red_pieces[1][1]) in action.get_next_game_state().get_rep().get_env():
                                    return action
                    else:
                        for action in current_state.generate_possible_stateful_actions():
                            if (blue_pieces[0][0], red_pieces[1][1]) in action.get_next_game_state().get_rep().get_env():
                                return action
                else:
                    center_pieces = [(i, j) for i in range(5, 9) for j in range(5, 9)]
                    random.shuffle(center_pieces)  # Melange les positions centrales
                    for (i, j) in center_pieces:
                        for action in current_state.generate_possible_stateful_actions():
                            if (i, j) in action.get_next_game_state().get_rep().get_env():
                                return action

        raise MethodNotImplementedError()
