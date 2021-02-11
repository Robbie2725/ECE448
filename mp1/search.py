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


def compute_mst_cost(waypoints, wp_bitmap_tuple):
    wp_bitmap = list(wp_bitmap_tuple)
    wp_list = []
    for i in range(len(wp_bitmap)):
        if wp_bitmap[i] == 0:
            wp_list.append(waypoints[i])

    size = len(wp_list)
    if size == 1 or size == 0:
        return 0

    g = Graph(size)
    for node1 in range(size):
        for node2 in range(size):
            if node2 == node1:
                continue  # skip edge if dest, source are the same node
            g.add_edge(node1, node2, heuristic(wp_list[node1], wp_list[node2]))

    cost = 0
    for edge in g.kruskal():
        cost += edge[2]
    return cost


def all_waypoints_reached(cur_state):
    cur_tuple = cur_state[1]
    for i in cur_tuple:
        if i == 0:
            return False
    return True


def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """

    # start search at maze start
    cur_cell = maze.start

    frontier = []  # initialize priority queue of (path_cost+heuristic, state)

    goal_list = []
    for i in maze.waypoints:
        goal_list.append(0)
    start_goal_tuple = tuple(goal_list)

    # Cache the MST cost of the remaining waypoints, initially the only entry is all waypoints
    mst_map = {start_goal_tuple: compute_mst_cost(maze.waypoints, start_goal_tuple)}

    # find the closest waypoint
    wp_d = []
    for i in maze.waypoints:
        wp_d.append(heuristic(cur_cell, i))

    start_state = (cur_cell, start_goal_tuple)
    cur_state = start_state

    # add state to frontier
    pq.heappush(frontier, (min(wp_d)+mst_map[start_goal_tuple], start_state))

    parents = {start_state: None}  # Maps current state -> parent state
    g_cost = {start_state: 0}  # Maps current state -> best path cost

    wp_list = list(maze.waypoints)
    while len(frontier) != 0:
        frontier_node = pq.heappop(frontier)
        cur_state = frontier_node[1]
        cur_cell = cur_state[0]
        # Check if current state is in the goal state
        if all_waypoints_reached(cur_state):
            break

        # add/update neighbors
        neighbors = maze.neighbors(cur_cell[0], cur_cell[1])
        for cell in neighbors:
            # check if cell is waypoint, if so then update the waypoint map
            if cell in maze.waypoints:
                old_goal_map = list(cur_state[1])
                old_goal_map[wp_list.index(cell)] = 1
                new_goal_tuple = tuple(old_goal_map)
            else:
                # else just use the goal state of the parent
                new_goal_tuple = cur_state[1]

            new_g = g_cost[cur_state] + 1
            new_state = (cell, new_goal_tuple)

            # if no g_cost, then add to frontier automatically
            if new_state not in g_cost.keys():
                # compute the heuristic
                if new_goal_tuple in mst_map.keys():
                    # if already computed, use the cached value
                    mst_cost = mst_map[new_goal_tuple]
                else:
                    # else compute the mst cost and save for later
                    mst_cost = compute_mst_cost(maze.waypoints, new_goal_tuple)
                    mst_map[new_goal_tuple] = mst_cost
                i = 0
                d_list = []
                for wp in maze.waypoints:
                    if new_goal_tuple[i] == 0:
                        i += 1
                        d_list.append(heuristic(cell, wp))
                    else:
                        i += 1
                        continue
                if len(d_list) == 0:
                    d_list = [0]
                pq.heappush(frontier, (new_g+min(d_list)+mst_cost, new_state))
                g_cost[new_state] = new_g
                parents[new_state] = cur_state
            elif new_g < g_cost[new_state]:
                # compute the heuristic
                if new_goal_tuple in mst_map.keys():
                    # if already computed, use the cached value
                    mst_cost = mst_map[new_goal_tuple]
                else:
                    # else compute the mst cost and save for later
                    mst_cost = compute_mst_cost(maze.waypoints, new_goal_tuple)
                    mst_map[new_goal_tuple] = mst_cost
                i = 0
                d_list = []
                for wp in maze.waypoints:
                    if new_goal_tuple[i] == 0:
                        i += 1
                        d_list.append(heuristic(cell, wp))
                    else:
                        i += 1
                        continue
                pq.heappush(frontier, (new_g+min(d_list) + mst_cost, new_state))
                g_cost[new_state] = new_g
                parents[new_state] = cur_state

    # reconstruct path
    path = [cur_cell]
    parent_state = parents[cur_state]
    while parents[parent_state] is not None:
        path.insert(0, parent_state[0])
        parent_state = parents[parent_state]
    path.insert(0, parent_state[0])
    return path


def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    # Algorithm for part 3 works for part 4 as well
    return astar_corner(maze)


def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []


# A graph class I found that implements kruskal's algorithm for finding an MST
# Source is at https://www.pythonpool.com/kruskals-algorithm-python/
# Instead of printing the result, I modified it to return the MST edges instead
class Graph:
    def __init__(self, vertex):
        self.V = vertex
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    def search(self, parent, i):
        if parent[i] == i:
            return i
        return self.search(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.search(parent, x)
        yroot = self.search(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskal(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.search(parent, u)
            y = self.search(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)
        return result
