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
    explored = {cur_cell: (None, 0)}  # maps ID -> (parent, cost+heuristic)

    # add all the neighbors of the start to the frontier
    for cell in maze.neighbors(cur_cell[0], cur_cell[1]):
        g_plus_h = 1 + heuristic(cell, maze_goal)
        pq.heappush(frontier, (g_plus_h, (cell, cur_cell)))

    # iterate to first node in frontier
    frontier_node = pq.heappop(frontier)
    cur_cell = frontier_node[1][0]

    # continue to iterate until goal is reached
    while cur_cell != maze_goal:

        # compute cost + heuristic for the current cell
        new_total_c = frontier_node[0]

        # if this cell has been explored, and the explored cost+heuristic is higher, then skip
        if cur_cell in explored.keys():
            old_total_c = explored[cur_cell][1]
            if new_total_c >= old_total_c:
                frontier_node = pq.heappop(frontier)
                cur_cell = frontier_node[1][0]
                continue

        explored[cur_cell] = (frontier_node[1][1], new_total_c)
        # add/update neighbors in frontier
        neighbors = maze.neighbors(cur_cell[0], cur_cell[1])
        for cell in neighbors:
            total_c = frontier_node[0]+1+heuristic(cell, maze_goal)
            pq.heappush(frontier, (total_c, (cell, cur_cell)))
        frontier_node = pq.heappop(frontier)
        cur_cell = frontier_node[1][0]

    path = [cur_cell]
    cur_parent = frontier_node[1][1]
    while cur_parent != maze.start:
        path.insert(0, cur_parent)
        explored_tuple = explored[cur_parent]
        cur_parent = explored_tuple[0]
    path.insert(0, cur_parent)

    return path


def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    return []


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
