# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

from time import sleep
import heapq as pq
from kruskal_check import Graph


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # vairables for search
    cur_cell = maze.start
    bfs_frontier = []  # list of (ID, parent, cost), managed with queue methods
    bfs_explored = {cur_cell: (None, 0)}  # maps ID -> (parent, cost)
    for cell in maze.neighbors(cur_cell[0], cur_cell[1]):
        bfs_frontier.append((cell, cur_cell, 1))
    frontier_cell = bfs_frontier.pop(0)
    cur_cell = frontier_cell[0]
    while cur_cell not in maze.waypoints:
        # check if cost higher than current cost in explored
        if cur_cell in bfs_explored.keys():
            if frontier_cell[2] >= bfs_explored[cur_cell][1]:
                frontier_cell = bfs_frontier.pop(0)
                cur_cell = frontier_cell[0]
                continue
        # if not in the explored set, add it (or update it if cost lower)
        bfs_explored[cur_cell] = (frontier_cell[1], frontier_cell[2])
        # add neighbors to frontier
        neighbors = maze.neighbors(cur_cell[0], cur_cell[1])
        for cell in neighbors:
            bfs_frontier.append((cell, cur_cell, frontier_cell[2] + 1))
        # iterate to next in frontier
        frontier_cell = bfs_frontier.pop(0)
        cur_cell = frontier_cell[0]

    # build the path from the parent nodes in the explored sets
    path = [cur_cell]
    cur_parent = frontier_cell[1]
    while cur_parent != maze.start:
        path.insert(0, cur_parent)
        explored_tuple = bfs_explored[cur_parent]
        cur_parent = explored_tuple[0]
    path.insert(0, cur_parent)

    return path


def heuristic(cur, goal):
    """
    Computes the manhattan distance between a given point and the goal
    @param cur: Point #1 (tuple)
    @param goal: Point #2 (tuple)
    @return distance: the distance b/w Point #1 and Point #2 (int)
    """
    return abs(cur[0] - goal[0]) + abs(cur[1] - goal[1])


def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # vairables for search
    cur_cell = maze.start
    maze_goal = maze.waypoints[0]
    frontier = []  # priority queue of (cost+heuristic, (node, parent))
    explored = {cur_cell: None}  # maps ID -> parent
    g_cost = {cur_cell: 0}  # the current best path cost to the cell

    pq.heappush(frontier, (heuristic(cur_cell, maze_goal), (cur_cell, None)))  # Add the start node to the frontier

    while len(frontier) != 0:
        #  remove first node in pq, if goal state then end and begin reconstruct path
        frontier_node = pq.heappop(frontier)
        cur_cell = frontier_node[1][0]
        if cur_cell == maze_goal:
            break

        # add/update neighbors in frontier
        neighbors = maze.neighbors(cur_cell[0], cur_cell[1])
        for cell in neighbors:
            new_g = g_cost[cur_cell] + 1
            #  if no current g cost, then automatically add
            if cell not in g_cost.keys():
                pq.heappush(frontier, (new_g + heuristic(cell, maze_goal), (cell, cur_cell)))
                g_cost[cell] = new_g
                explored[cell] = cur_cell

            # if there is a g cost, then only add if the current one is better
            if new_g < g_cost[cell]:
                g_cost[cell] = new_g
                pq.heappush(frontier, (new_g + heuristic(cell, maze_goal), (cell, cur_cell)))
                explored[cell] = cur_cell

    #  reconstruct path
    path = [cur_cell]
    cur_parent = explored[cur_cell]
    while cur_parent != maze.start:
        path.insert(0, cur_parent)
        cur_parent = explored[cur_parent]
    path.insert(0, cur_parent)

    return path


def compute_mst_cost(wp_list, wp_bitmap):
    size = 0
    for i in range(len(wp_bitmap)):
        if wp_bitmap[i] == 0:
            size += 1

    if size == 1:
        return 0

    g = Graph(size)
    for node1 in range(len(wp_bitmap)):
        if wp_bitmap[node1] == 1:
            continue
        for node2 in range(len(wp_bitmap)):
            if node2 == node1 or wp_bitmap[node2] == 1:
                continue  # skip edge if dest, source are the same node
            g.add_edge(node1, node2, heuristic(wp_list[node1], wp_list[node2]))
            # pq.heappush(edges, (heuristic(waypoints[node1], waypoints[node2]), node1, node2))  # sorted list of edges
    # e = 0
    # print(edges)
    # # number of edges in MST is = # nodes - 1
    # while e < len(waypoints)-1:
    #     cur_edge = pq.heappop(edges)
    #     if (cur_edge[2], cur_edge[0]) in kruskal_tree[cur_edge[1]]:
    #         continue  # skip if they are already in same tree
    #     e += 1
    #     kruskal_tree[cur_edge[1]].append((cur_edge[2], cur_edge[0]))
    #     kruskal_tree[cur_edge[2]].append((cur_edge[1], cur_edge[0]))

    # cost = 0
    # print(kruskal_tree)
    # for key in kruskal_tree:
    #     cost += edge[1]
    cost = 0
    for edge in g.kruskal():
        cost += edge[2]
    return cost

def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    return []
    # run A* on each pair of start -> waypoint and waypoint -> waypoint
    # s_path_list = []
    # for wp in maze.waypoints:
    #     cur_cell = maze.start
    #     maze_goal = wp
    #     frontier = []  # priority queue of (cost+heuristic, (node, parent))
    #     explored = {cur_cell: (None, 0)}  # maps ID -> (parent, cost+heuristic)
    #
    #     # add all the neighbors of the start to the frontier
    #     for cell in maze.neighbors(cur_cell[0], cur_cell[1]):
    #         g_plus_h = 1 + heuristic(cell, maze_goal)
    #         pq.heappush(frontier, (g_plus_h, (cell, cur_cell)))
    #
    #     # iterate to first node in frontier
    #     frontier_node = pq.heappop(frontier)
    #     cur_cell = frontier_node[1][0]
    #
    #     # continue to iterate until goal state is reached
    #     while cur_cell != maze_goal:
    #
    #         # compute cost + heuristic for the current cell
    #         new_total_c = frontier_node[0]
    #
    #         # if this cell has been explored, and the explored cost+heuristic is higher, then skip
    #         if cur_cell in explored.keys():
    #             old_total_c = explored[cur_cell][1]
    #             if new_total_c >= old_total_c:
    #                 frontier_node = pq.heappop(frontier)
    #                 cur_cell = frontier_node[1][0]
    #                 continue
    #
    #         explored[cur_cell] = (frontier_node[1][1], new_total_c)
    #         # add/update neighbors in frontier
    #         neighbors = maze.neighbors(cur_cell[0], cur_cell[1])
    #         for cell in neighbors:
    #             total_c = frontier_node[0] + 1 + heuristic(cell, maze_goal)
    #             pq.heappush(frontier, (total_c, (cell, cur_cell)))
    #         frontier_node = pq.heappop(frontier)
    #         cur_cell = frontier_node[1][0]
    #
    #     path = [cur_cell]
    #     path_c = frontier_node[0]
    #     cur_parent = frontier_node[1][1]
    #     while cur_parent != maze.start:
    #         path.insert(0, cur_parent)
    #         explored_tuple = explored[cur_parent]
    #         cur_parent = explored_tuple[0]
    #     path.insert(0, cur_parent)
    #
    #     s_path_list.append(path)
    #
    # return []
    # # vairables for search
    # cur_cell = maze.start
    # frontier = []  # priority queue of (path_cost+heuristic, state)
    #
    # wp_list = []
    # goals_left = []
    # for wp in maze.waypoints:
    #     wp_list.append(wp)
    #     goals_left.append(0)
    # start_state = (cur_cell, maze.waypoints)  # state = (location, waypoints left)
    # explored = {start_state: (None, 0)}  # maps state -> (parent_state, cost)
    #
    # mst_map = {}
    #
    # # add all the neighbors of the start to the frontier
    # for cell in maze.neighbors(cur_cell[0], cur_cell[1]):
    #     # check if waypoint, update goal state if so
    #     if cell in maze.waypoints:
    #         wp_list =
    #     new_state = (cell, new_goals)
    #     # find heuristic
    #     i = 0
    #     d_list = []
    #     for wp in wp_list:
    #         if new_goals[i] == 1:
    #             i += 1
    #             continue
    #         d_list.append(heuristic(cell, wp_list[i]))
    #         i += 1
    #     mst_cost = compute_mst_cost(wp_list, new_goals)
    #     mst_map[new_goals] = mst_cost
    #     h_cost = mst_cost + min(d_list)
    #     pq.heappush(frontier, (1 + h_cost, new_state))
    #
    # print(frontier)
    # return []


def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []


def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
