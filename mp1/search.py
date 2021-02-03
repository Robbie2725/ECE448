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

from help import *

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # vairables for search
    columns, rows = get_size(maze)
    cur_cell = maze.start
    print(cur_cell)
    bfs_frontier = []  # list of (ID, parent, cost)
    bfs_explored = {cur_cell: (None, 0)}  # maps ID -> (parent, cost)
    for cell in maze.neighbors(cur_cell[0], cur_cell[1]):
        bfs_frontier.append((cell, cur_cell, 1))
    frontier_cell = (cur_cell, None, 0)
    while frontier_cell[0] != maze.legend.waypoint:
        # print(bfs_frontier)
        # check if cost lower than current cost in explored
        if frontier_cell[0] in bfs_explored.keys():
            if frontier_cell[2] < bfs_explored[frontier_cell[0]][1]:
                bfs_frontier[frontier_cell[0]] = (frontier_cell[1], frontier_cell[2])
            else:
                continue
        # add neighbors to frontier
        neighbors = maze.neighbors(frontier_cell[0])
        # print(neighbors)
        for cell in neighbors:
            bfs_frontier.append((cell, frontier_cell[0], frontier_cell[2]+1))
        frontier_cell = bfs_frontier.pop(0)

    path = [frontier_cell[1]]
    cur_parent = frontier_cell[1]
    while cur_parent != maze.start:
        path.index(0, cur_parent)
    path.index(0, cur_parent)

    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []

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
    
            
