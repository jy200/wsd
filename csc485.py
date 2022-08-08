from collections import defaultdict
class Graph:


    def __init__(self, vertices):

        self.M = vertices   # Total number of vertices in the graph

        self.graph = []     # Array of edges



    # Add edges

    def add_edge(self, a, b, c):

        self.graph.append([a, b, c])



    # Print the solution

    def print_solution(self, distance):

        print("Vertex Distance from Source")

        for k in range(self.M):

            print("{0}\t\t{1}".format(k+1, distance[k]))



    def bellman_ford(self, src):



        distance = [float("Inf")] * self.M

        distance[src] = 0

        hop_dict = defaultdict()

        for i in range(self.M - 1):

            for a, b, c in self.graph:

                if distance[a] != float("Inf") and distance[a] + c < distance[b]:
                    distance[b] = distance[a] + c
                    hop_dict[b+1] = [distance[b], a+1]


            # for index, dis in enumerate(distance):
            #     print(index+1, dis)
            # sorted(hop_dict)
            # print(hop_dict)
            print(sorted(hop_dict.items()))


        # for a, b, c in self.graph:
        #
        #     if distance[a] != float("Inf") and distance[a] + c < distance[b]:
        #
        #         print("Graph contains negative weight cycle")
        #
        #         return



        self.print_solution(distance)



# g = Graph(5)
# g.add_edge(0, 1, 2)
# g.add_edge(0, 2, 4)
# g.add_edge(1, 3, 2)
# g.add_edge(2, 4, 3)
# g.add_edge(2, 3, 4)
# g.add_edge(4, 3, -5)

# g = Graph(8)
# g.add_edge(1, 2, 3)
# g.add_edge(1, 3, 2)
# g.add_edge(2, 3, 2)
# g.add_edge(2, 4, 7)
# g.add_edge(4,6,1)
# g.add_edge(6,8,8)
# g.add_edge(3,5,4)
# g.add_edge(5,7,2)
# g.add_edge(7,8,2)
# g.add_edge(2,5,3)
# g.add_edge(3,4,1)
# g.add_edge(4,5,1)
# g.add_edge(4,7,8)
# g.add_edge(5,6,1)
# g.add_edge(6,7,1)
g = Graph(8)
g.add_edge(0, 1, 3)
g.add_edge(0, 2, 2)
g.add_edge(1, 2, 2)
g.add_edge(1, 3, 7)
g.add_edge(3,5,1)
g.add_edge(5,7,8)
g.add_edge(2,4,4)
g.add_edge(4,6,2)
g.add_edge(6,7,2)
g.add_edge(1,4,3)
g.add_edge(2,3,1)
g.add_edge(3,4,1)
g.add_edge(3,6,8)
g.add_edge(4,5,1)
g.add_edge(5,6,1)
g.bellman_ford(0)

# a = "1000"
# b = "1000"
# y = int(a,2) ^ int(b,2)
# print('{0:b}'.format(y))

# y = int("000000000",2) ^ int("11100001",2) ^ int("10011001",2) ^ int("01111000",2)
# print('{0:b}'.format(y))
